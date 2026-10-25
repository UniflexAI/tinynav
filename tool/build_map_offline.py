#!/usr/bin/env python3
"""Offline map build by direct bag parsing: one sequential pass replaces the
BagPlayer -> looper_bridge -> DDS -> 4-way-sync chain of build_map_node.py.

Contract messages are deserialized in-process; keyframes are gated with the
bridge's own rule (translation / rotation / idle); rgb is matched nearest
within the builder's 20 ms slop; the REAL BuildMapNode.process() runs per
keyframe on messages constructed via the same fast paths the bridge uses live
(cv2_to_imgmsg / np2msg). Only the transport disappears -- the processing
semantics are the replay path's, stage timers included.

Progress is reported as `MAPPING_PERCENT:<pct>` stdout lines (the same lines
build_map_node's logger emits), read by pilot's build tee. The percentage
comes from the bag's metadata.yaml message count; without it no lines are
emitted except the final 100.

Two product-neutral overlaps (both parity-checked, parity_check.py 14 items):
- viz suppression: the trajectory / local-map / TF publishes feed only rviz;
  no product artifact reads them. TINYNAV_BUILD_VIZ=1 restores them for debug.
- media single-writer: depth/infra1/rgb VideoDB writes queue to ONE writer
  thread in FIFO order == keyframe timestamp order, so the encoded files are
  byte-identical to sequential writes. The depths shelve stays SYNCHRONOUS:
  detect_loop_closure reads the current keyframe's depth immediately.

Storage-agnostic: --storage follows rosbag2_py plugin ids (sqlite3 / mcap).
"""
import argparse
import collections
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, os.environ.get("TINYNAV_ROOT", "/tinynav"))

from tinynav.core.build_map_node import BuildMapNode
from tinynav.core.math_utils import np2msg

DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
INFRA_TOPIC = "/camera/camera/infra1/image_rect_raw"
VIO_IMAGE_TOPIC = "/camera/camera/vio_image"
COLOR_TOPIC = "/camera/camera/color/image_rect_raw/compressed"
INFRA2_CAMINFO = "/camera/camera/infra2/camera_info"
COLOR_CAMINFO = "/camera/camera/color/camera_info"
TF_STATIC = "/tf_static"

COLOR_SLOP_NS = 20_000_000  # the builder 4-way sync slop for the rgb queue

_PERCENT_INTERVAL_S = 2.0


def stamp_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def pose_to_T(pose) -> np.ndarray:
    p, q = pose.position, pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def rot_angle(T_prev: np.ndarray, T_cur: np.ndarray) -> float:
    dR = T_prev[:3, :3].T @ T_cur[:3, :3]
    return float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))


def bag_message_count(bag_file: str, storage: str) -> int | None:
    """The bag's total message count, from metadata.yaml beside it. A `.db3`
    uri keeps its metadata.yaml in the parent directory. None: report no
    percentage and fall back to the final 100 only."""
    directory = bag_file if os.path.isdir(bag_file) else os.path.dirname(
        os.path.abspath(bag_file))
    try:
        with open(os.path.join(directory, 'metadata.yaml')) as fh:
            for line in fh:
                if 'message_count' in line:
                    return int(line.split(':')[1].strip())
    except (OSError, ValueError):
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument("--bag_file", required=True,
                    help="rosbag2 directory or .db3 file")
    ap.add_argument("--map_save_path", required=True)
    ap.add_argument("--storage", default="sqlite3", help="rosbag2 plugin id")
    ap.add_argument("--keyframe-translation", type=float, default=0.03)
    ap.add_argument("--keyframe-rotation-deg", type=float, default=1.0)
    ap.add_argument("--idle-keyframe-s", type=float, default=3.0)
    args = ap.parse_args()
    rot_gate = np.deg2rad(args.keyframe_rotation_deg)

    total_msgs = bag_message_count(args.bag_file, args.storage)
    last_percent_t = 0.0

    def report_percent(done: int) -> None:
        nonlocal last_percent_t
        if total_msgs is None:
            return
        now = time.monotonic()
        if done and now - last_percent_t < _PERCENT_INTERVAL_S:
            return
        last_percent_t = now
        print(f"MAPPING_PERCENT:{min(99.9, 100.0 * done / total_msgs):.1f}",
              flush=True)

    rclpy.init()
    node = BuildMapNode(args.map_save_path, verbose_timer=True)
    bridge = CvBridge()

    if os.environ.get('TINYNAV_BUILD_VIZ', '') != '1':
        # These three publishes feed only rviz/foxglove; no product artifact
        # reads them. Timer blocks remain, so the suppressed tax is still
        # visible in the stage table.
        node.pose_graph_trajectory_publish = lambda timestamp: None
        node.publish_local_map = lambda point_cloud, frame_id: None
        node.publish_all_transforms = lambda: None

    # Media single-writer: only the VideoDB encodes (infra1/rgb) are queued --
    # nothing reads them back until save_mapping, which runs after the queue
    # is drained. The depths shelve stays synchronous (detect_loop_closure
    # reads the current keyframe's depth immediately). Embeddings / features /
    # patch_tokens stay synchronous too: loop closure and VLAD read them
    # during mapping.
    real_db = node.db

    class MediaWriteProxy:
        def __init__(self, real, q):
            self._real = real
            self._q = q

        def set_entry(self, key, **kw):
            media = {k: kw.pop(k) for k in ("infra1_image", "rgb_image") if k in kw}
            if media:
                self._q.put((key, media))
            if kw:
                self._real.set_entry(key, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

    media_q = queue.Queue(maxsize=64)

    def media_writer():
        while True:
            item = media_q.get()
            try:
                if item is None:
                    return
                key, kw = item
                real_db.set_entry(key, **kw)
            finally:
                media_q.task_done()

    writer_thread = threading.Thread(target=media_writer, daemon=True)
    writer_thread.start()
    node.db = MediaWriteProxy(real_db, media_q)

    # Single decoder thread: color frames are decoded ahead of the reader
    # (cv2 releases the GIL) into dec_cache; the MAIN thread is the only one
    # moving them out of it, so matching/eviction stays single-threaded. A
    # cache miss decodes synchronously -- the fallback, not the design.
    decode_q = queue.Queue(maxsize=16)
    dec_cache = {}

    def decode_color(msg):
        arr = bridge.compressed_imgmsg_to_cv2(msg)
        img = bridge.cv2_to_imgmsg(arr, encoding="bgr8")
        img.header = msg.header
        return img

    def color_decoder():
        while True:
            item = decode_q.get()
            try:
                if item is None:
                    return
                stamp, msg = item
                dec_cache[stamp] = decode_color(msg)
                while len(dec_cache) > 96:
                    dec_cache.pop(min(dec_cache))
            finally:
                decode_q.task_done()

    decoder_thread = threading.Thread(target=color_decoder, daemon=True)
    decoder_thread.start()

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=args.bag_file, storage_id=args.storage),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cls = {name: get_message(t) for name, t in types.items()}

    triplets = {}  # stamp_ns -> {"vio": PoseStamped, "depth": Image, "infra": Image}
    infra2_done = color_caminfo_done = False
    # (stamp_ns, raw CompressedImage); the decoder stays <=16 ahead, so a frame
    # needed within the 20 ms slop can never be evicted from the ring's tail.
    color_ring = collections.deque(maxlen=40)
    last_kf_T = None
    last_kf_stamp_s = None
    n_kf = 0
    n_dropped_colorless = 0
    n_read = 0
    stats = {"read": 0.0, "process": 0.0}
    t_wall = time.monotonic()

    def handle_keyframe(stamp, parts):
        nonlocal last_kf_T, last_kf_stamp_s, n_kf, n_dropped_colorless
        pose_msg = parts[VIO_IMAGE_TOPIC]
        T = pose_to_T(pose_msg.pose)
        stamp_s = stamp / 1e9
        is_kf = (
            last_kf_T is None
            or np.linalg.norm(T[:3, 3] - last_kf_T[:3, 3]) >= args.keyframe_translation
            or rot_angle(last_kf_T, T) >= rot_gate
            or stamp_s - last_kf_stamp_s > args.idle_keyframe_s
        )
        if not is_kf:
            return
        rgb_msg = None
        if color_ring:
            best = min(color_ring, key=lambda e: abs(e[0] - stamp))
            if abs(best[0] - stamp) <= COLOR_SLOP_NS:
                s = best[0]
                rgb_msg = dec_cache.pop(s, None)
                if rgb_msg is None:
                    rgb_msg = decode_color(best[1])  # decoder lagged; decode now
        if rgb_msg is None:
            n_dropped_colorless += 1
            if n_dropped_colorless <= 3:
                span = (f"[{color_ring[0][0]}..{color_ring[-1][0]}] n={len(color_ring)}"
                        if color_ring else "EMPTY")
                print(f"DBG colorless kf={stamp} ring={span} kf_n={n_kf}", flush=True)
            return  # the builder's 4-way sync drops colorless keyframes likewise

        t0 = time.monotonic()
        stamp_msg = pose_msg.header.stamp
        depth_m = bridge.imgmsg_to_cv2(parts[DEPTH_TOPIC], desired_encoding="passthrough")
        depth_m = np.asarray(depth_m)
        if parts[DEPTH_TOPIC].encoding in ("mono16", "16UC1"):
            depth_m = depth_m.astype(np.float32) / 1000.0
        else:
            depth_m = depth_m.astype(np.float32)
        depth_out = bridge.cv2_to_imgmsg(depth_m, encoding="32FC1")
        depth_out.header.stamp = stamp_msg
        depth_out.header.frame_id = "camera"

        image_out = parts[INFRA_TOPIC]  # driver owns it; re-stamp in place
        image_out.header.stamp = stamp_msg
        image_out.header.frame_id = "camera"

        odom_msg = np2msg(T, stamp_msg, "world", "camera")
        node.process(image_out, odom_msg, depth_out, rgb_msg)
        stats["process"] += time.monotonic() - t0
        last_kf_T, last_kf_stamp_s, n_kf = T, stamp_s, n_kf + 1

    while reader.has_next():
        topic, raw, _ts = reader.read_next()
        cls = msg_cls.get(topic)
        if cls is None:
            continue
        n_read += 1
        report_percent(n_read)
        t0 = time.monotonic()
        if topic in (DEPTH_TOPIC, INFRA_TOPIC, VIO_IMAGE_TOPIC):
            msg = deserialize_message(raw, cls)
            triplets.setdefault(stamp_ns(msg.header.stamp), {})[topic] = msg
            for s in [s for s, p in triplets.items() if len(p) == 3]:
                handle_keyframe(s, triplets.pop(s))
            if len(triplets) > 8:  # prune stale partials, keep the newest
                for s in sorted(triplets)[:-4]:
                    del triplets[s]
        elif topic == INFRA2_CAMINFO and not infra2_done:
            node.info_callback(deserialize_message(raw, cls))
            infra2_done = True
        elif topic == COLOR_CAMINFO and not color_caminfo_done:
            node.rgb_camera_info_callback(deserialize_message(raw, cls))
            color_caminfo_done = True
        elif topic == TF_STATIC:
            node.tf_callback(deserialize_message(raw, cls))
        elif topic == COLOR_TOPIC:
            msg = deserialize_message(raw, cls)
            s = stamp_ns(msg.header.stamp)
            color_ring.append((s, msg))
            decode_q.put((s, msg))
        # /camera/camera/imu, infra1 camera_info and the remaining bag topics
        # have no consumer here: the bridge carried them for live subscribers
        # the offline build does not have.
        stats["read"] += time.monotonic() - t0

    report_percent(total_msgs or 0)
    if total_msgs is not None:
        print("MAPPING_PERCENT:100.0", flush=True)

    # Order matters: the decoder drains before the writer (a queued color
    # frame's process() may still enqueue media), the writer before save.
    # Queue.join must pair with task_done -- a join without it blocks forever.
    decode_q.join()
    decode_q.put(None)
    decoder_thread.join()
    media_q.join()
    media_q.put(None)
    writer_thread.join()

    t_save = time.monotonic()
    node.save_mapping()
    print(
        f"[build_map_offline] keyframes={n_kf} colorless_dropped={n_dropped_colorless} "
        f"messages={n_read} read={stats['read']:.1f}s process={stats['process']:.1f}s "
        f"save={time.monotonic() - t_save:.1f}s wall={time.monotonic() - t_wall:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

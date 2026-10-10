"""Offline evaluation of the stair mode target generator on a rosbag.

Ground truth is where the robot actually went: for every depth frame we compare the
direction to the generated target with the direction of the recorded trajectory 1 m ahead.
Every raw pose is fed to the generator so its odometry jump check sees the full-rate stream.

usage:
  source /opt/ros/humble/setup.bash
  .venv/bin/python tool/stair_offline_eval.py --bag tinynav_db/ros2bags/bag_downstairs --direction down --out output/stair_eval
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tinynav.core.stair_memory import StairMemory, prior_direction  # noqa: E402
from tinynav.core.stair_target import StairConfig, StairTargetGenerator  # noqa: E402

GT_ARC = 1.0


def read_bag(bag, depth_topic, pose_topic, image_topic, info_topic):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[depth_topic, pose_topic, image_topic, info_topic]))
    def stamp(h):
        return h.stamp.sec + h.stamp.nanosec * 1e-9
    depth, images, poses, K = [], [], [], None
    while r.has_next():
        topic, data, _ = r.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == pose_topic:
            pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
            p, q = pose.position, pose.orientation
            poses.append([stamp(msg.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        elif topic == info_topic and K is None:
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        elif topic == depth_topic:
            if msg.encoding in ('16UC1', 'mono16'):
                depth.append((stamp(msg.header), np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width), 1e-3))
            else:
                depth.append((stamp(msg.header), np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width), 1.0))
        elif topic == image_topic:
            images.append((stamp(msg.header), np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., 0]))
    return depth, images, np.array(poses), K


def pose_at(poses, t):
    j = int(np.clip(np.searchsorted(poses[:, 0], t), 1, len(poses) - 1))
    j = j if abs(poses[j, 0] - t) < abs(poses[j - 1, 0] - t) else j - 1
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(poses[j, 4:8]).as_matrix()
    T[:3, 3] = poses[j, 1:4]
    return T, j, abs(poses[j, 0] - t)


def future_point(poses, j, arc):
    step = np.linalg.norm(np.diff(poses[j:, 1:3], axis=0), axis=1)
    k = np.searchsorted(np.cumsum(step), arc)
    return poses[j + k + 1, 1:4] if k < len(step) else None


def phase_labeler(poses, window=1.0, min_dz=0.3):
    """flight: recorded height changes within +-window s; landing: flat between the first and last flight."""
    t, z = poses[:, 0], poses[:, 3]
    lo = np.interp(t - window, t, z)
    hi = np.interp(t + window, t, z)
    flight_t = t[np.abs(hi - lo) > min_dz]
    def phase_of(ts):
        if len(flight_t) and np.min(np.abs(flight_t - ts)) < 0.05:
            return 'flight'
        return 'landing' if len(flight_t) and flight_t[0] < ts < flight_t[-1] else 'outside'
    return phase_of


def angle_deg(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def render(res, cfg, T, poses, j, image, t_rel, err):
    scale = 5
    height = res['height']
    n = height.shape[0]
    ground_z = T[2, 3] - cfg.camera_height
    rel = np.clip((np.nan_to_num(height, nan=ground_z) - ground_z) / 1.5, -1, 1)
    vis = cv2.applyColorMap(((rel + 1) * 127.5).astype(np.uint8), cv2.COLORMAP_COOL)
    vis[~np.isfinite(height)] = (35, 35, 35)
    if 'reachable' in res:
        vis[res['reachable']] = (0.55 * vis[res['reachable']] + 0.45 * np.array([90, 200, 90])).astype(np.uint8)
    vis[res['obstacle']] = (200, 200, 200)
    vis = cv2.resize(vis.transpose(1, 0, 2)[::-1], (n * scale, n * scale), interpolation=cv2.INTER_NEAREST)
    def to_px(xy):
        return int((xy[0] - res['origin'][0]) / cfg.resolution * scale), int(n * scale - (xy[1] - res['origin'][1]) / cfg.resolution * scale)
    fut = poses[j:j + 400:10, 1:3]
    for a, b in zip(fut[:-1], fut[1:]):
        cv2.line(vis, to_px(a), to_px(b), (255, 255, 255), 2)
    if res['path'] is not None:
        pts = [to_px(p) for p in res['path'][:, :2]]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(vis, a, b, (0, 140, 255), 2)
    if res['target'] is not None:
        cv2.circle(vis, to_px(res['target']), 9, (0, 0, 255) if res['status'] == 'ok' else (0, 200, 255), -1)
    c = to_px(T[:2, 3])
    fwd = T[:2, :3] @ np.array([0, 0, 1.0])
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    cv2.circle(vis, c, 7, (0, 255, 255), -1)
    cv2.arrowedLine(vis, c, to_px(T[:2, 3] + 0.5 * fwd), (0, 255, 255), 2, tipLength=0.3)
    txt = f"t={t_rel:5.1f}s  {res['status']}  well={res.get('well_side', 0):+.1f}" + (f"  err={err:4.0f}deg" if err is not None else "")
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if res['status'] in ('odom_invalid', 'no_seed'):
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 8)
        cv2.putText(vis, 'STOP', (vis.shape[1] // 2 - 70, vis.shape[0] // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
    img = cv2.cvtColor(cv2.resize(image, (int(image.shape[1] * n * scale / image.shape[0]), n * scale)), cv2.COLOR_GRAY2BGR)
    return np.hstack([img, vis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--direction', choices=['up', 'down'], required=True)
    ap.add_argument('--out', default='output/stair_eval')
    ap.add_argument('--depth-topic', default='/camera/camera/depth/image_rect_raw')
    ap.add_argument('--pose-topic', default='/camera/camera/vio_100hz')
    ap.add_argument('--image-topic', default='/camera/camera/infra1/image_rect_raw')
    ap.add_argument('--info-topic', default='/camera/camera/infra1/camera_info')
    ap.add_argument('--memory', type=float, default=None, help='override StairConfig.memory_s')
    ap.add_argument('--rate', type=float, default=None, help='target update rate in Hz (default: every depth frame); '
                    'frames in between reuse the last target, odom_invalid is still immediate')
    ap.add_argument('--turn', choices=['auto', 'left', 'right'], default='auto', help='U-turn side at landings (auto: estimate)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a StairConfig field, repeatable')
    ap.add_argument('--stair-memory', default=None, help='stair memory (tool/stair_memory.py build) used as a direction prior')
    ap.add_argument('--features', default=None, help='image features of this bag (tool/stair_memory.py features), needed with --stair-memory')
    ap.add_argument('--stop-at-landing', action='store_true',
                    help="stay stopped after 'landing' (default: start stair mode again, like pressing Stairs on the landing)")
    ap.add_argument('--no-video', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    depth, images, poses, K = read_bag(args.bag, args.depth_topic, args.pose_topic, args.image_topic, args.info_topic)
    print(f"depth {len(depth)}, images {len(images)}, poses {len(poses)}")
    overrides = {k: float(v) for k, v in (kv.split('=', 1) for kv in args.set)}
    if args.memory is not None:
        overrides['memory_s'] = args.memory
    cfg = StairConfig(**overrides)
    memory = feat_t = feat_f = None
    if args.stair_memory:
        if not args.features:
            raise SystemExit('--stair-memory needs --features for the evaluated bag')
        memory = StairMemory(args.stair_memory)
        feats = np.load(args.features)
        feat_t, feat_f = feats['t'], feats['f']
        print(f"memory: {len(memory)} frames")
    phase_of = phase_labeler(poses)
    gen = StairTargetGenerator(cfg)
    gen.turn_side = {'auto': 0, 'left': 1, 'right': -1}[args.turn]
    img_t = np.array([t for t, _ in images])
    t0 = depth[0][0]
    writer, rows, next_pose = None, [], 0
    res, last_compute, compute_times = None, -np.inf, []
    for t, d, depth_scale in depth:
        while next_pose < len(poses) and poses[next_pose, 0] <= t:
            gen.add_pose(poses[next_pose, 0], poses[next_pose, 1:4])
            next_pose += 1
        T, j, sync_err = pose_at(poses, t)
        if sync_err > 0.05:
            continue
        gen.add_depth(t, d.astype(np.float32) * depth_scale, K, T)
        due = args.rate is None or t - last_compute >= 1.0 / args.rate
        if res is None or due or not gen.odom_valid or res['status'] == 'odom_invalid':
            tic = time.perf_counter()
            prior, mem_sim = None, np.nan
            if memory is not None:
                q = int(np.argmin(np.abs(feat_t - t)))
                if abs(feat_t[q] - t) < 0.1:
                    bearing, mem_sim = memory.query(feat_f[q], args.direction)
                    prior = None if bearing is None else prior_direction(T, bearing)
            res = gen.compute(T, args.direction, prior_dir=prior)
            if res['status'] == 'landing' and not args.stop_at_landing:
                gen.new_run()
            compute_times.append(time.perf_counter() - tic)
            last_compute = t
        fut = future_point(poses, j, GT_ARC)
        err = None
        if res['target'] is not None and fut is not None:
            err = angle_deg(res['target'][:2] - T[:2, 3], fut[:2] - T[:2, 3])
        rows.append(dict(t=t - t0, phase=phase_of(t), status=res['status'], err_deg=err, cam_z=T[2, 3],
                         target_z=None if res['target'] is None else res['target'][2],
                         cam_x=T[0, 3], cam_y=T[1, 3], fwd_x=T[0, 2], fwd_y=T[1, 2],
                         target_x=None if res['target'] is None else res['target'][0],
                         target_y=None if res['target'] is None else res['target'][1],
                         guided=bool(res.get('guided', False)), mem_sim=mem_sim,
                         uturn=bool(res.get('uturn', False)), turn_in_place=bool(res.get('turn_in_place', False)),
                         landings=gen.landings, wall_keep=bool(res.get('wall_keep', False))))
        if not args.no_video:
            frame = render(res, cfg, T, poses, j, images[int(np.argmin(np.abs(img_t - t)))][1], t - t0, err)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(args.out, 'stair_eval.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    if writer is not None:
        writer.release()
    with open(os.path.join(args.out, 'stair_eval.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    status = np.array([r['status'] for r in rows])
    counted = [r['t'] for a, r in zip(rows[:-1], rows[1:]) if r['landings'] > a['landings']]
    print(f"landings counted: {gen.landings} at " + ' '.join(f'{x:.0f}s' for x in counted)
          + f" | camera z drop over the run {rows[0]['cam_z'] - min(r['cam_z'] for r in rows):.2f} m")
    ct = np.array(compute_times[20:]) * 1000  # skip numba warmup
    if memory is not None:
        print(f"memory direction used in {np.mean([r['guided'] for r in rows]) * 100:.0f}% of frames")
    print(f"compute calls {len(compute_times)} ({len(compute_times) / (rows[-1]['t'] - rows[0]['t']):.1f}/s), "
          f"median {np.median(ct):.1f} ms, p95 {np.percentile(ct, 95):.1f} ms")
    print(f"frames {len(rows)}: " + ", ".join(f"{s}={np.sum(status == s)}" for s in ('ok', 'search', 'no_seed', 'odom_invalid', 'landing')))
    invalid_t = np.array([r['t'] for r in rows if r['status'] == 'odom_invalid'])
    if len(invalid_t):
        breaks = np.where(np.diff(invalid_t) > 0.5)[0]
        spans = zip(np.r_[invalid_t[0], invalid_t[breaks + 1]], np.r_[invalid_t[breaks], invalid_t[-1]])
        print("odom_invalid (stop) spans: " + ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in spans))
    print("direction error vs recorded motion (odom_invalid frames have no target and are excluded):")
    for ph in ('flight', 'landing', 'outside'):
        sel = [r for r in rows if r['phase'] == ph and r['status'] != 'odom_invalid']
        errs = np.array([r['err_deg'] for r in sel if r['err_deg'] is not None])
        if len(errs):
            print(f"  {ph:8s} frames={len(sel):4d} with_target={len(errs) / len(sel) * 100:5.1f}%  median={np.median(errs):5.1f}deg  "
                  f"p90={np.percentile(errs, 90):5.1f}deg  >45deg={np.mean(errs > 45) * 100:5.1f}%")

if __name__ == '__main__':
    main()

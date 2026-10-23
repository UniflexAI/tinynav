"""Stream a ros2 bag and save synced infra1/infra2/depth frames at a fixed rate, plus the full-rate pose track.

Output dir: {name}_left.png, {name}_right.png, {name}_depth.png (uint16 mm), meta.json, poses.npy
poses.npy rows: [stamp_ns, x, y, z, qx, qy, qz, qw], camera(optical)-to-world, world z-up.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

NS = '/camera/camera'
LEFT, RIGHT, DEPTH = f'{NS}/infra1/image_rect_raw', f'{NS}/infra2/image_rect_raw', f'{NS}/depth/image_rect_raw'
INFO1, INFO2 = f'{NS}/infra1/camera_info', f'{NS}/infra2/camera_info'
POSE_TOPICS = (f'{NS}/vio_100hz', f'{NS}/vio_image', '/insight/vio_100hz', '/insight/vio_20hz',
               '/slam/odometry', '/slam/odometry_visual')  # first available wins
SYNC_NS = 20_000_000


def stamp(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def to_np(msg):
    dtype = np.uint16 if msg.encoding in ('16UC1', 'mono16') else np.uint8
    return np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)


def nearest(buf, s):
    if not buf:
        return None
    k = min(buf, key=lambda x: abs(x - s))
    return buf[k] if abs(k - s) < SYNC_NS else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--hz', type=float, default=2.0, help='saved frame rate')
    ap.add_argument('--baseline', type=float, default=0.0, help='override when infra2 camera_info has no baseline')
    ap.add_argument('--no_right', action='store_true', help='left + depth only (frames labelled from other teacher frames)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bufs = {LEFT: OrderedDict(), RIGHT: OrderedDict(), DEPTH: OrderedDict()}
    pending = OrderedDict()  # anchor stamp -> waiting for partners
    poses, info, frames = [], {}, []
    last_anchor = -1e18
    with AnyReader([Path(args.bag)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        topics = {c.topic for c in reader.connections}
        pose_topic = next((t for t in POSE_TOPICS if t in topics), None)
        anchor = DEPTH if DEPTH in topics else LEFT  # depth is the sparsest stream
        wanted = {LEFT, DEPTH, INFO1, INFO2, pose_topic} | (set() if args.no_right else {RIGHT})
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, _, raw in reader.messages(connections=conns):
            if conn.topic in (INFO1, INFO2):
                if conn.topic not in info:
                    msg = reader.deserialize(raw, conn.msgtype)
                    info[conn.topic] = dict(K=list(msg.k), P=list(msg.p), width=msg.width, height=msg.height)
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            s = stamp(msg)
            if conn.topic == pose_topic:
                pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose  # Odometry vs PoseStamped
                p, q = pose.position, pose.orientation
                poses.append([s, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                continue
            bufs[conn.topic][s] = msg
            if conn.topic == anchor and s - last_anchor >= 1e9 / args.hz:
                pending[s] = True
                last_anchor = s
            for a in list(pending):
                l = nearest(bufs[LEFT], a)
                r = l if args.no_right else nearest(bufs[RIGHT], a)
                if l is not None and r is not None and stamp(l) == stamp(r):
                    name = f'{len(frames):06d}'
                    cv2.imwrite(str(out / f'{name}_left.png'), to_np(l))
                    if not args.no_right:
                        cv2.imwrite(str(out / f'{name}_right.png'), to_np(r))
                    if anchor == DEPTH:
                        cv2.imwrite(str(out / f'{name}_depth.png'), to_np(bufs[DEPTH][a]))
                    frames.append(dict(name=name, stamp=int(stamp(l))))
                    pending.pop(a)
                elif s - a > 500_000_000:
                    pending.pop(a)
                    last_anchor = -1e18  # take the next anchor instead
            for buf in bufs.values():
                while buf and next(iter(buf)) < s - 1_000_000_000:
                    buf.popitem(last=False)

    poses = np.array(poses, dtype=np.float64).reshape(-1, 8)
    poses = poses[np.argsort(poses[:, 0])]
    i1, i2 = info[INFO1], info.get(INFO2)
    meta = dict(bag=str(Path(args.bag).resolve()), K=i1['K'], width=i1['width'], height=i1['height'],
                fx=i1['K'][0], baseline=args.baseline or (-i2['P'][3] / i2['P'][0] if i2 else 0.0), pose_topic=pose_topic, hz=args.hz, frames=frames)
    assert meta['baseline'] > 0.01, 'no stereo baseline in camera_info, pass --baseline'
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    np.save(out / 'poses.npy', poses)
    print(f'{Path(args.bag).name}: {len(frames)} frames, {len(poses)} poses ({pose_topic}), baseline={meta["baseline"]:.4f}')


if __name__ == '__main__':
    main()

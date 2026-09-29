"""Build the stair mode visual memory from teleoperated rosbags.

1. features: DINOv2 global feature of every image (needs TensorRT, i.e. the robot or the dev container)
     python tool/stair_memory.py features --bag <bag> --out <bag>_features.npz
2. build: label each frame with the direction the robot went over the next metre and merge bags (numpy only)
     python tool/stair_memory.py build --bag A --features A_features.npz --direction down \\
                                       --bag B --features B_features.npz --direction up --out stair_memory.npz
Use recordings of a person driving the robot normally through the stairwell the robot will work in.
"""
import argparse
import asyncio
import os
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tinynav.core.stair_memory import DIRECTIONS, label_frames  # noqa: E402

POSE_TOPICS = ('/camera/camera/vio_100hz', '/camera/camera/vio_image')


def open_bag(bag, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    return r, types


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def read_poses(bag):
    for topic in POSE_TOPICS:  # prefer the 100 Hz VIO, fall back to the image-rate one
        r, types = open_bag(bag, [topic])
        poses = []
        while r.has_next():
            _, data, _ = r.read_next()
            msg = deserialize_message(data, get_message(types[topic]))
            pose = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
            p, q = pose.position, pose.orientation
            poses.append([stamp(msg.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        if poses:
            return np.array(poses), topic
    raise SystemExit(f'{bag}: no pose topic among {POSE_TOPICS}')


def cmd_features(args):
    from tinynav.core.models_trt import Dinov2TRT
    model = Dinov2TRT(args.engine) if args.engine else Dinov2TRT()
    r, types = open_bag(args.bag, [args.image_topic])
    ts, feats = [], []
    while r.has_next():
        _, data, _ = r.read_next()
        msg = deserialize_message(data, get_message(types[args.image_topic]))
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[..., 0]
        ts.append(stamp(msg.header))
        feats.append(np.asarray(asyncio.run(model.infer(img)), dtype=np.float32))
    np.savez(args.out, t=np.array(ts), f=np.stack(feats).astype(np.float16))
    print(f'{args.bag}: {len(ts)} image features -> {args.out}')


def cmd_build(args):
    if not (len(args.bag) == len(args.features) == len(args.direction)):
        raise SystemExit('give one --features and one --direction per --bag')
    feats, bearing, direction, source = [], [], [], []
    for i, (bag, fpath, d) in enumerate(zip(args.bag, args.features, args.direction)):
        poses, topic = read_poses(bag)
        f = np.load(fpath)
        labels = label_frames(f['t'], poses, ahead=args.ahead)
        keep = np.isfinite(labels)
        feats.append(f['f'][keep])
        bearing.append(labels[keep])
        direction.append(np.full(keep.sum(), DIRECTIONS[d], np.int8))
        source.append(np.full(keep.sum(), i, np.int16))
        print(f'{bag} ({d}, poses from {topic}): kept {keep.sum()} / {len(keep)} frames')
    np.savez(args.out, features=np.concatenate(feats).astype(np.float16), bearing=np.concatenate(bearing),
             direction=np.concatenate(direction), source=np.concatenate(source), bags=np.array(args.bag))
    print(f'memory: {sum(len(b) for b in bearing)} frames -> {args.out}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('features', help='DINOv2 features of every image in a bag (needs TensorRT)')
    f.add_argument('--bag', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--image-topic', default='/camera/camera/infra1/image_rect_raw')
    f.add_argument('--engine', default=None, help='DINOv2 TensorRT engine (default: the one map_node uses)')
    b = sub.add_parser('build', help='label frames and merge bags into a memory file')
    b.add_argument('--bag', action='append', required=True)
    b.add_argument('--features', action='append', required=True)
    b.add_argument('--direction', action='append', required=True, choices=list(DIRECTIONS))
    b.add_argument('--ahead', type=float, default=1.0, help='label = direction of the point this far ahead [m]')
    b.add_argument('--out', required=True)
    args = ap.parse_args()
    cmd_features(args) if args.cmd == 'features' else cmd_build(args)


if __name__ == '__main__':
    main()

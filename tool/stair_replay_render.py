"""Render a stair replay recording (tool/stair_replay_recorder.py) into a video and print stats.

Left: camera image. Right: planning's obstacle map with the planning trajectory (green), stair target
(red) and stair path (orange), the recorded motion (white) and the robot (yellow). A red frame with
STOP marks frames where stair_node told planning to stop. The replay is open loop: the robot follows
the recording, so this shows what planning would choose at each moment.

usage:
  .venv/bin/python tool/stair_replay_render.py --bag <bag> --rec rec.pkl --out output/stair_replay/stair_replay.mp4
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stair_offline_eval import GT_ARC, angle_deg, future_point, phase_labeler, pose_at, read_bag  # noqa: E402

SCALE = 6


def latest(rec, key, t):
    ts = np.array([e[0] for e in rec[key]]) if rec[key] else np.zeros(0)
    i = np.searchsorted(ts, t, side='right') - 1
    return rec[key][i] if i >= 0 else None


def draw_line(vis, pts, color, width):
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(vis, a, b, color, width)


def render_frame(rec, poses, t, t_rel, image):
    T, j, _ = pose_at(poses, t)
    m = latest(rec, 'mask', t)
    if m is None:
        return None, None
    _, org, res, bits, shape = m
    mask = np.unpackbits(bits)[:shape[0] * shape[1]].reshape(shape).astype(bool)
    W = shape[0] * SCALE
    vis = np.full((shape[0], shape[1], 3), 45, np.uint8)
    vis[mask] = (190, 190, 190)
    vis = cv2.resize(vis.transpose(1, 0, 2)[::-1], (W, W), interpolation=cv2.INTER_NEAREST)

    def px(xy):
        return int((xy[0] - org[0]) / res * SCALE), int(W - (xy[1] - org[1]) / res * SCALE)

    draw_line(vis, [px(p) for p in poses[j:j + 400:10, 1:3]], (255, 255, 255), 2)
    target, stop = latest(rec, 'target', t), latest(rec, 'stop', t)
    stopped = target is None or (stop is not None and stop[0] > target[0])
    if not stopped:
        path = latest(rec, 'stair_path', t)
        if path is not None and len(path[1]) > 1:
            draw_line(vis, [px(p) for p in path[1][:, :2]], (0, 140, 255), 2)
        cv2.circle(vis, px(target[1]), 9, (0, 0, 255), -1)
    plan = latest(rec, 'plan', t)
    has_plan = plan is not None and t - plan[0] < 0.3 and len(plan[1]) > 1
    err = None
    if has_plan:
        draw_line(vis, [px(p) for p in plan[1][:, :2]], (0, 255, 0), 4)
        fut = future_point(poses, j, GT_ARC)
        if fut is not None and np.linalg.norm(plan[1][-1, :2] - T[:2, 3]) > 0.2:
            err = angle_deg(plan[1][-1, :2] - T[:2, 3], fut[:2] - T[:2, 3])
    fwd = T[:2, :3] @ np.array([0, 0, 1.0])
    fwd /= np.linalg.norm(fwd) + 1e-9
    cv2.circle(vis, px(T[:2, 3]), 7, (0, 255, 255), -1)
    cv2.arrowedLine(vis, px(T[:2, 3]), px(T[:2, 3] + 0.5 * fwd), (0, 255, 255), 2, tipLength=0.3)
    status = latest(rec, 'status', t)
    txt = f"t={t_rel:5.1f}s stair:{status[1].split()[1] if status else '-'} planning:{'path' if has_plan else 'NO PATH'}"
    cv2.putText(vis, txt + (f" err={err:3.0f}deg" if err is not None else ''), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    if stopped:
        cv2.rectangle(vis, (0, 0), (W - 1, W - 1), (0, 0, 255), 8)
        cv2.putText(vis, 'STOP', (W // 2 - 70, W // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
    img = cv2.cvtColor(cv2.resize(image, (int(image.shape[1] * W / image.shape[0]), W)), cv2.COLOR_GRAY2BGR)
    return np.hstack([img, vis]), (stopped, has_plan, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--rec', required=True, help='pickle written by stair_replay_recorder.py')
    ap.add_argument('--out', default='output/stair_replay/stair_replay.mp4')
    ap.add_argument('--depth-topic', default='/camera/camera/depth/image_rect_raw')
    ap.add_argument('--pose-topic', default='/camera/camera/vio_100hz')
    ap.add_argument('--image-topic', default='/camera/camera/infra1/image_rect_raw')
    ap.add_argument('--info-topic', default='/camera/camera/infra1/camera_info')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.rec, 'rb') as f:
        rec = pickle.load(f)
    depth, images, poses, _ = read_bag(args.bag, args.depth_topic, args.pose_topic, args.image_topic, args.info_topic)
    img_t = np.array([t for t, _ in images])
    phase_of = phase_labeler(poses)
    t0 = depth[0][0]
    writer, stats = None, []
    for t, _, _ in depth:
        frame, s = render_frame(rec, poses, t, t - t0, images[int(np.argmin(np.abs(img_t - t)))][1])
        if frame is None:
            continue
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        stats.append((phase_of(t), *s))
    if writer is not None:
        writer.release()
    print("planning trajectory vs recorded motion 1 m ahead (frames where stair_node has a target):")
    for ph in ('flight', 'landing', 'outside'):
        sel = [s for s in stats if s[0] == ph and not s[1]]
        errs = np.array([s[3] for s in sel if s[3] is not None])
        if sel:
            print(f"  {ph:8s} frames={len(sel):4d} planning has path={np.mean([s[2] for s in sel]) * 100:5.1f}%" + (
                f"  median={np.median(errs):5.1f}deg p90={np.percentile(errs, 90):5.1f}deg >45deg={np.mean(errs > 45) * 100:4.1f}%" if len(errs) else ''))
    print(f"stopped frames: {sum(s[1] for s in stats)}")


if __name__ == '__main__':
    main()

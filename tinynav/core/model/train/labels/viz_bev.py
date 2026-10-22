"""Visualize BEV labels next to the image and a planning-style baseline from single-frame camera depth.

Panels: infra1 | baseline (planning rule on camera depth, this frame only) | label classes | label ground height
Classes: gray unknown, green free, red obstacle (span), orange step/edge, magenta railing, blue outline traversed,
white dots = cells seen by this frame alone (the rest was filled in from neighbouring frames).
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .make_bev_gt import BAND, BEV_N, heading_frame, load_frames, to_grid

SPAN_OBS, STEP_OBS = 0.15, 0.12          # label rule (wheeled default)
PLAN_BAND, PLAN_SPAN = (-0.4, 0.4), 0.2  # planning_node ObstacleConfig: band rel. camera, min_wall_span


def classes(lab):
    known, span, step = lab['known'], lab['span'].astype(np.float32), lab['step'].astype(np.float32)
    cls = np.zeros(known.shape, np.uint8)            # 0 unknown
    cls[known] = 1                                   # 1 free
    cls[known & (step >= STEP_OBS)] = 4              # 4 step / edge (stairs, curbs, drops)
    cls[known & (span >= SPAN_OBS)] = 2              # 2 obstacle (walls, boxes)
    cls[lab['rail']] = 3                             # 3 railing
    return cls


def baseline(depth_mm, T, K, cam_h):
    """planning_node.build_obstacle_map on one frame of camera depth (z-span inside the camera-relative band)."""
    z = depth_mm[::2, ::2].astype(np.float32) / 1000.0
    v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]] * 2
    ok = (z > 0.2) & (z < 4.5)
    pc = np.stack([(u[ok] - K[2]) / K[0] * z[ok], (v[ok] - K[5]) / K[4] * z[ok], z[ok]], 1)
    pw = pc @ T[:3, :3].T + T[:3, 3]
    Rb, origin = heading_frame(T, cam_h)
    pb = (pw - origin) @ Rb.T
    r, q, k, okg = to_grid(pb)
    zr = pb[:, 2] - cam_h                          # relative to camera height, like planning_node
    sel = okg & (zr >= PLAN_BAND[0]) & (zr <= PLAN_BAND[1])
    hi = np.full((BEV_N, BEV_N), -9.0)
    lo = np.full((BEV_N, BEV_N), 9.0)
    np.maximum.at(hi, (r[sel], q[sel]), zr[sel])
    np.minimum.at(lo, (r[sel], q[sel]), zr[sel])
    seen = np.zeros((BEV_N, BEV_N), bool)
    seen[r[okg], q[okg]] = True
    cls = np.where(seen, 1, 0).astype(np.uint8)
    cls[(hi - lo) >= PLAN_SPAN] = 2
    return cls


COLORS = np.array([[90, 90, 90], [60, 170, 90], [220, 50, 40], [230, 60, 230], [250, 160, 30]], np.uint8)


def render(cls, lab=None):
    img = COLORS[cls].copy()
    if lab is not None:
        edge = lab['traversed'] & ~cv2.erode(lab['traversed'].astype(np.uint8), np.ones((3, 3))).astype(bool)
        img[edge] = (60, 120, 255)
        vis = lab['visible'] & (cls > 0)
        img[vis] = (0.6 * img[vis] + 0.4 * 255).astype(np.uint8)
    img = cv2.resize(img, (BEV_N * 3, BEV_N * 3), interpolation=cv2.INTER_NEAREST)
    c = (BEV_N * 3 // 2, int(BEV_N * 3 * 4.8 / 6.4))   # camera at x=0, y=0
    cv2.circle(img, c, 5, (255, 255, 0), -1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dir')
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--out', default=None)
    ap.add_argument('--bev', default='bev')
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    d = Path(args.frames_dir)
    meta, _, frames = load_frames(d)
    info = json.loads((d / args.bev / 'info.json').read_text())
    out = Path(args.out or d / 'viz_bev')
    out.mkdir(parents=True, exist_ok=True)
    for f in frames[::args.every]:
        if args.only and f['name'] not in args.only:
            continue
        lab = dict(np.load(d / args.bev / f"{f['name']}.npz"))
        img = cv2.imread(str(d / f"{f['name']}_left.png"), cv2.IMREAD_GRAYSCALE)
        dp = d / f"{f['name']}_depth.png"
        base = baseline(cv2.imread(str(dp), cv2.IMREAD_UNCHANGED), f['T'], meta['K'], info['cam_h']) \
            if dp.exists() else np.zeros((BEV_N, BEV_N), np.uint8)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5.6))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title(f"{d.name} {f['name']}", fontsize=9)
        ax[1].imshow(render(base))
        ax[1].set_title('baseline: planning rule, camera depth, this frame', fontsize=9)
        ax[2].imshow(render(classes(lab), lab))
        ax[2].set_title('label (gray unknown/green free/red obstacle/orange step/magenta rail)', fontsize=9)
        g = np.where(lab['known'], lab['ground'].astype(np.float32), np.nan)
        im = ax[3].imshow(cv2.resize(g, (BEV_N * 3, BEV_N * 3), interpolation=cv2.INTER_NEAREST),
                          cmap='terrain', vmin=BAND[0], vmax=0.6)
        ax[3].set_title('label ground height (m, rel. ground under camera)', fontsize=9)
        fig.colorbar(im, ax=ax[3], shrink=0.7)
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        fig.savefig(out / f"{f['name']}.png", dpi=70)
        plt.close(fig)
    print('saved to', out)


if __name__ == '__main__':
    main()

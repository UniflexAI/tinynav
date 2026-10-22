"""Side-by-side: image | camera depth | FoundationStereo depth | segmentation, plus top-down views of both depths."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RAIL_WORDS = ('railing', 'rail', 'bannister', 'balustrade', 'fence', 'handrail')


def rail_ids(labels):
    return [int(k) for k, v in labels.items() if any(w in v.lower() for w in RAIL_WORDS)]


def topdown(ax, depth, K, mask=None, title='', zmax=5.0, step=2):
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[::step, ::step]
    ok = (z > 0.1) & (z < zmax)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    sc = ax.scatter(x[ok], z[ok], c=-y[ok], s=0.3, cmap='viridis', vmin=-1.0, vmax=1.0)
    if mask is not None:
        m = ok & mask[::step, ::step]
        ax.scatter(x[m], z[m], c='red', s=0.6)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, zmax)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    return sc


def color_depth(d, dmax=5.0):
    vis = cv2.applyColorMap((np.clip(d / dmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[d <= 0] = 0
    return vis[..., ::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    d = Path(args.frames_dir)
    out = Path(args.out or d / 'viz')
    out.mkdir(exist_ok=True)
    meta = json.loads((d / 'meta.json').read_text())
    labels = json.loads((d / 'seg_labels.json').read_text())
    rails = rail_ids(labels)
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, (256, 3)).astype(np.uint8)
    K = meta['K']

    for f in meta['frames']:
        n = f['name']
        if args.only and n not in args.only or not (d / f'{n}_fsdepth.npy').exists():
            continue
        left = cv2.imread(str(d / f'{n}_left.png'), cv2.IMREAD_GRAYSCALE)
        fs = np.load(d / f'{n}_fsdepth.npy')
        seg = cv2.imread(str(d / f"{n}_seg.png"), cv2.IMREAD_UNCHANGED)
        cam_path = d / f'{n}_depth.png'
        cam = cv2.imread(str(cam_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 if cam_path.exists() else np.zeros_like(fs)
        if cam.shape != fs.shape:
            cam = cv2.resize(cam, fs.shape[::-1], interpolation=cv2.INTER_NEAREST)
        rail = np.isin(seg, rails)

        seg_vis = (0.5 * np.repeat(left[..., None], 3, -1) + 0.5 * palette[seg]).astype(np.uint8)
        seg_vis[rail] = (255, 0, 0)
        present = [labels[str(i)] for i in np.unique(seg) if (seg == i).mean() > 0.01]

        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.4])
        for i, (img, title) in enumerate([(left, 'infra1'), (color_depth(cam), 'camera depth'),
                                          (color_depth(fs), 'FoundationStereo depth'),
                                          (seg_vis, 'seg (red = railing/fence)')]):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        topdown(fig.add_subplot(gs[1, 0:2]), cam, K, rail, 'top-down: camera depth (red = railing pixels)')
        sc = topdown(fig.add_subplot(gs[1, 2:4]), fs, K, rail, 'top-down: FoundationStereo depth (red = railing pixels)')
        fig.colorbar(sc, ax=fig.axes[-1], label='height rel. camera (m)', shrink=0.7)
        fig.suptitle(f'{n}  classes: {", ".join(present)[:200]}', fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f'{n}.png', dpi=80)
        plt.close(fig)
        print('saved', out / f'{n}.png', flush=True)


if __name__ == '__main__':
    main()

"""Run BevNet on extracted frames (left + camera depth + pose) and save per-frame obstacle probabilities.

  python -m tinynav.core.model.train.predict --ckpt runs/bev_v1/best.pt <frames_dir>

Writes <frames_dir>/pred/{frame}.npz with obs_<preset> (P(observed and obstacle), float16) and known.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .bev_geometry import cam_to_bev_rotation, depth_raster, sampling_grid
from .cost import PRESETS, obstacle_prob
from .dataset import IMG_HW, scale_K
from .model import BevNet, split_heads
from .pack import DEPTH_DS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('frames_dir')
    args = ap.parse_args()
    d = Path(args.frames_dir)
    meta = json.loads((d / 'meta.json').read_text())
    poses = np.load(d / 'poses.npy')
    K = np.array(meta['K'], np.float64).reshape(3, 3)
    Kd = K.copy()
    Kd[:2] /= DEPTH_DS
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BevNet(pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)['model'])
    out = d / 'pred'
    out.mkdir(exist_ok=True)
    for f in meta['frames']:
        j = int(np.argmin(np.abs(poses[:, 0] - f['stamp'])))
        R_cb = cam_to_bev_rotation(R.from_quat(poses[j, 4:8]).as_matrix())
        img = cv2.imread(str(d / f"{f['name']}_left.png"), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(d / f"{f['name']}_depth.png"), cv2.IMREAD_UNCHANGED)[::DEPTH_DS, ::DEPTH_DS].astype(np.float32) / 1000
        Ks = scale_K(K, IMG_HW[1] / img.shape[1], IMG_HW[0] / img.shape[0])
        x = cv2.resize(img, IMG_HW[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        grid, valid = sampling_grid(Ks, R_cb, IMG_HW)
        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None].to(dev)
        o = split_heads(model(t(x[None]), t(depth_raster(depth, Kd, R_cb, step=1)), t(grid), t(valid)))
        o = {k: torch.sigmoid(v[0]).float().cpu().numpy() for k, v in o.items() if k != 'ground'}
        np.savez_compressed(out / f"{f['name']}.npz", known=o['known'].astype(np.float16),
                            **{f'obs_{p}': obstacle_prob(o['span'], o['step'], o['rail'], P).astype(np.float16)
                               for p, P in PRESETS.items()})
    print(f'{d.name}: {len(meta["frames"])} predictions')


if __name__ == '__main__':
    main()

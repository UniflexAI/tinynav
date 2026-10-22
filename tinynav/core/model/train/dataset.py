"""Training samples: runtime inputs (infra1 + camera depth + camera tilt) -> teacher BEV labels.

index_bags() reads label folders produced by labels/ (meta.json, poses.npy, {name}_left.png, {name}_depth.png,
bev/{name}.npz, bev/info.json); pack.py turns them into the compact files BevDataset reads.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from .bev_geometry import cam_to_bev_rotation, depth_raster, sampling_grid

IMG_HW = (320, 272)          # network input, half of the 640x544 infra1 image
F_CAM = np.diag([-1.0, 1.0, 1.0])
F_BEV = np.diag([1.0, -1.0, 1.0])


def scale_K(K, sx, sy):
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 1] *= sy
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def index_bags(root, bags=None):
    """[(bag_dir, frame_name, R_cam_to_world, cam_h, K)] for frames with label + camera depth."""
    items = []
    for d in sorted(Path(root).iterdir()):
        info_p = d / 'bev' / 'info.json'
        if (bags and d.name not in bags) or not info_p.exists():
            continue
        meta = json.loads((d / 'meta.json').read_text())
        cam_h = json.loads(info_p.read_text())['cam_h']
        poses = np.load(d / 'poses.npy')
        K = np.array(meta['K'], np.float64).reshape(3, 3)
        for f in meta['frames']:
            if not ((d / 'bev' / f"{f['name']}.npz").exists() and (d / f"{f['name']}_depth.png").exists()):
                continue
            j = int(np.argmin(np.abs(poses[:, 0] - f['stamp'])))
            items.append((d, f['name'], R.from_quat(poses[j, 4:8]).as_matrix(), cam_h, K))
    return items


class BevDataset(Dataset):
    """Reads samples written by pack.py: <packed_dir>/<bag>/<frame>.npz"""

    def __init__(self, root, bags, augment=False):
        self.files = sorted(f for b in bags for f in (Path(root) / b).glob('*.npz'))
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        img, depth = z['img'], z['depth'].astype(np.float32) / 1000.0
        H, W = z['img_hw']
        K = scale_K(z['K'], IMG_HW[1] / W, IMG_HW[0] / H)
        Kd = z['K_depth'].copy()
        R_cb = cam_to_bev_rotation(z['R_cw'])
        layers = {k: z[k] for k in ('known', 'ground', 'span', 'step', 'rail', 'visible')}
        if self.augment and np.random.rand() < 0.5:
            img, depth = img[:, ::-1], depth[:, ::-1]
            K = K.copy()
            K[0, 2] = img.shape[1] - 1 - K[0, 2]
            Kd[0, 2] = depth.shape[1] - 1 - Kd[0, 2]
            R_cb = F_BEV @ R_cb @ F_CAM
            layers = {k: v[:, ::-1] for k, v in layers.items()}
        raster = depth_raster(np.ascontiguousarray(depth), Kd, R_cb, step=1)
        img = img.astype(np.float32) / 255
        if self.augment:
            img = np.clip(img * np.random.uniform(0.6, 1.4) + np.random.uniform(-0.1, 0.1), 0, 1) ** np.random.uniform(0.7, 1.4)
        grid, valid = sampling_grid(K, R_cb, IMG_HW)
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        known = layers['known'].astype(bool)
        return dict(
            img=f32(img[None]), raster=f32(raster), grid=f32(grid), valid=f32(valid),
            known=f32(known), visible=f32(layers['visible']),
            ground=f32(np.where(known, layers['ground'].astype(np.float32) - float(z['cam_h']), 0)),
            span=f32(layers['span']), step=f32(layers['step']), rail=f32(layers['rail']),
        )

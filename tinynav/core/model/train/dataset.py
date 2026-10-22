"""Training samples: runtime inputs (infra1 + camera depth + camera tilt) -> teacher BEV labels.

Expects bag folders produced by labels/extract_frames.py, labels/run_teachers.py and labels/make_bev_gt.py:
  meta.json, poses.npy, {name}_left.png, {name}_depth.png, bev/{name}.npz, bev/info.json
"""
import json
from pathlib import Path

import cv2
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
    def __init__(self, root, bags=None, augment=False):
        self.items = index_bags(root, bags)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, name, R_cw, cam_h, K = self.items[i]
        img = cv2.imread(str(d / f'{name}_left.png'), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(d / f'{name}_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        lab = np.load(d / 'bev' / f'{name}.npz')
        R_cb = cam_to_bev_rotation(R_cw)
        layers = {k: lab[k] for k in ('known', 'ground', 'span', 'step', 'rail', 'visible')}
        if self.augment and np.random.rand() < 0.5:
            img, depth = img[:, ::-1], depth[:, ::-1]
            K = K.copy()
            K[0, 2] = img.shape[1] - 1 - K[0, 2]
            R_cb = F_BEV @ R_cb @ F_CAM
            layers = {k: v[:, ::-1] for k, v in layers.items()}
        raster = depth_raster(depth, K, R_cb, step=4)
        Ks = scale_K(K, IMG_HW[1] / img.shape[1], IMG_HW[0] / img.shape[0])
        img = cv2.resize(np.ascontiguousarray(img), IMG_HW[::-1], interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.augment:
            img = np.clip(img * np.random.uniform(0.6, 1.4) + np.random.uniform(-0.1, 0.1), 0, 1) ** np.random.uniform(0.7, 1.4)
        grid, valid = sampling_grid(Ks, R_cb, IMG_HW)
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        known = layers['known'].astype(bool)
        return dict(
            img=f32(img[None]), raster=f32(raster), grid=f32(grid), valid=f32(valid),
            known=f32(known), visible=f32(layers['visible']),
            ground=f32(np.where(known, layers['ground'].astype(np.float32) - cam_h, 0)),
            span=f32(layers['span']), step=f32(layers['step']), rail=f32(layers['rail']),
        )

"""Training samples: runtime inputs (infra1 + camera depth + camera tilt) -> teacher BEV labels.

index_bags() reads label folders produced by labels/ (meta.json, poses.npy, {name}_left.png, {name}_depth.png,
bev/{name}.npz, bev/info.json); pack.py turns them into the compact files BevDataset reads.
"""
import json
from pathlib import Path

import numpy as np
import torch
import cv2
from scipy.ndimage import binary_closing, grey_dilation
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from .bev_geometry import IMG_HW, cam_to_bev_rotation, depth_raster, sampling_grid, scale_K

F_CAM = np.diag([-1.0, 1.0, 1.0])
F_BEV = np.diag([1.0, -1.0, 1.0])


GAP_KERNEL = (5, 3)  # rows x cols; far teacher depth leaves gaps between rows of observed cells


def fill_gaps(layers):
    """Close small holes in the label (stripes between far depth rows) so 'known' is learnable."""
    known = layers['known'].astype(bool)
    closed = binary_closing(known, structure=np.ones(GAP_KERNEL, bool))
    new = closed & ~known
    out = dict(layers, known=closed)
    for k in ('ground', 'span', 'step'):
        v = layers[k].astype(np.float32)
        out[k] = np.where(new, grey_dilation(np.where(known, v, -9.0), size=GAP_KERNEL), v)
    out['rail'] = layers['rail'].astype(bool) | (new & grey_dilation(layers['rail'].astype(np.uint8), size=GAP_KERNEL).astype(bool))
    return out


def augment_image(img, K, rng):
    """Random zoom-in crop (intrinsics follow), noise and blur. img float32 (H,W) in [0,1]."""
    H, W = img.shape
    s = rng.uniform(1.0, 1.15)
    h, w = int(round(H / s)), int(round(W / s))
    y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
    img = cv2.resize(img[y0:y0 + h, x0:x0 + w], (W, H), interpolation=cv2.INTER_LINEAR)
    K = K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    K = scale_K(K, W / w, H / h)
    img = np.clip(img * rng.uniform(0.6, 1.4) + rng.uniform(-0.1, 0.1), 0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    img = np.clip(img + rng.normal(0, rng.uniform(0, 0.03), img.shape), 0, 1)
    return img.astype(np.float32), K


def augment_depth(depth, rng):
    """Holes and range-dependent noise like a real stereo camera (depth in meters, 0 = invalid)."""
    depth = depth * (1 + rng.normal(0, 0.01, depth.shape) * np.minimum(depth, 6.0))
    H, W = depth.shape
    for _ in range(rng.integers(0, 4)):
        h, w = rng.integers(H // 10, H // 3), rng.integers(W // 10, W // 3)
        y, x = rng.integers(0, H - h), rng.integers(0, W - w)
        depth[y:y + h, x:x + w] = 0
    return depth.astype(np.float32)


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

    def sample_weights(self, gain=4.0, full=0.02):
        """1 for frames without obstacles up to 1 + gain once obstacles cover `full` of the near box."""
        cache = {}
        w = []
        for f in self.files:
            if f.parent not in cache:
                p = f.parent / 'index.json'
                cache[f.parent] = json.loads(p.read_text()) if p.exists() else {}
            w.append(1.0 + gain * min(cache[f.parent].get(f.stem, 0.0) / full, 1.0))
        return w

    def __getitem__(self, i):
        z = np.load(self.files[i])
        img, depth = z['img'], z['depth'].astype(np.float32) / 1000.0
        H, W = z['img_hw']
        K = scale_K(z['K'], IMG_HW[1] / W, IMG_HW[0] / H)
        Kd = z['K_depth'].copy()
        R_cb = cam_to_bev_rotation(z['R_cw'])
        layers = fill_gaps({k: z[k] for k in ('known', 'ground', 'span', 'step', 'rail', 'visible')})
        if self.augment and np.random.rand() < 0.5:
            img, depth = img[:, ::-1], depth[:, ::-1]
            K = K.copy()
            K[0, 2] = img.shape[1] - 1 - K[0, 2]
            Kd[0, 2] = depth.shape[1] - 1 - Kd[0, 2]
            R_cb = F_BEV @ R_cb @ F_CAM
            layers = {k: v[:, ::-1] for k, v in layers.items()}
        img = np.ascontiguousarray(img, dtype=np.float32) / 255
        if self.augment:
            rng = np.random.default_rng()
            img, K = augment_image(img, K, rng)
            depth = augment_depth(np.ascontiguousarray(depth), rng)
        raster = depth_raster(np.ascontiguousarray(depth), Kd, R_cb, step=1)
        grid, valid = sampling_grid(K, R_cb, IMG_HW)
        def f32(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        known = layers['known'].astype(bool)
        return dict(
            img=f32(img[None]), raster=f32(raster), grid=f32(grid), valid=f32(valid),
            known=f32(known), visible=f32(layers['visible']),
            ground=f32(np.where(known, layers['ground'].astype(np.float32) - float(z['cam_h']), 0)),
            span=f32(layers['span']), step=f32(layers['step']), rail=f32(layers['rail']),
        )

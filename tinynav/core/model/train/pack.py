"""Pack label folders into compact per-sample files for training (small enough to copy to a training box).

Each sample keeps only what training reads: infra1 at network resolution, camera depth subsampled by DEPTH_DS,
camera rotation, intrinsics, camera height and the BEV label layers.

  python -m tinynav.core.model.train.pack <data_dir> <packed_dir>
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .dataset import IMG_HW, index_bags

DEPTH_DS = 4
LAYERS = ('known', 'ground', 'span', 'step', 'rail', 'visible')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    args = ap.parse_args()
    out = Path(args.out)
    items = index_bags(args.data)
    done = 0
    for d, name, R_cw, cam_h, K in items:
        dst = out / d.name / f'{name}.npz'
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(d / f'{name}_left.png'), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(d / f'{name}_depth.png'), cv2.IMREAD_UNCHANGED)
        lab = np.load(d / 'bev' / f'{name}.npz')
        # slicing keeps pixel centres at multiples of DEPTH_DS, so intrinsics just divide
        Kd = K.copy()
        Kd[:2] /= DEPTH_DS
        np.savez_compressed(
            dst, img=cv2.resize(img, IMG_HW[::-1], interpolation=cv2.INTER_AREA),
            img_hw=np.array(img.shape), depth=depth[::DEPTH_DS, ::DEPTH_DS], K=K, K_depth=Kd,
            R_cw=R_cw, cam_h=np.float32(cam_h), **{k: lab[k] for k in LAYERS})
        done += 1
    print(f'packed {done} new samples ({len(items)} total) into {out}')


if __name__ == '__main__':
    main()

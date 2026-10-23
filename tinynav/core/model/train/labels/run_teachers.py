"""Run teacher models on extracted frames: FoundationStereo depth + ADE20K semantic segmentation."""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch

FS_DIR = Path(os.environ.get('FOUNDATION_STEREO_DIR', Path.home() / 'workspace/teachers/FoundationStereo'))
# FoundationStereo's Utils imports open3d/trimesh only for point-cloud export; we don't need them.
for mod in ('open3d', 'trimesh'):
    try:
        __import__(mod)
    except ImportError:
        sys.modules[mod] = types.ModuleType(mod)
sys.path.insert(0, str(FS_DIR))


def load_stereo(ckpt):
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    cfg = OmegaConf.load(Path(ckpt).parent / 'cfg.yaml')
    cfg.setdefault('vit_size', 'vitl')
    cfg['valid_iters'] = 32
    model = FoundationStereo(cfg)
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    return model.cuda().eval()


@torch.no_grad()
def run_stereo(model, left, right, iters=32):
    from core.utils.utils import InputPadder
    def t(im):
        return torch.as_tensor(np.repeat(im[..., None], 3, -1)).cuda().float()[None].permute(0, 3, 1, 2)
    a, b = t(left), t(right)
    padder = InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = padder.pad(a, b)
    with torch.autocast('cuda', dtype=torch.float16):
        disp = model.forward(a, b, iters=iters, test_mode=True)
    return padder.unpad(disp.float())[0, 0].cpu().numpy()


def load_seg(name):
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    proc = AutoImageProcessor.from_pretrained(name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(name).cuda().eval()
    return proc, model


@torch.no_grad()
def run_seg(proc, model, gray):
    rgb = np.repeat(gray[..., None], 3, -1)
    inputs = proc(images=rgb, return_tensors='pt').to('cuda')
    out = model(**inputs)
    return proc.post_process_semantic_segmentation(out, target_sizes=[gray.shape])[0].cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dir')
    ap.add_argument('--ckpt', default=str(FS_DIR / 'pretrained_models/23-51-11/model_best_bp2.pth'))
    ap.add_argument('--seg', default='facebook/mask2former-swin-large-ade-semantic')
    ap.add_argument('--only', nargs='*', help='frame names to process (default: all)')
    ap.add_argument('--camera_depth', action='store_true',
                    help='no right image: use the camera depth as teacher depth (weaker labels), still run segmentation')
    args = ap.parse_args()
    d = Path(args.frames_dir)
    meta = json.loads((d / 'meta.json').read_text())
    frames = [f for f in meta['frames'] if not args.only or f['name'] in args.only]

    stereo = None if args.camera_depth else load_stereo(args.ckpt)
    proc, seg = load_seg(args.seg)
    labels = seg.config.id2label
    (d / 'seg_labels.json').write_text(json.dumps({int(k): v for k, v in labels.items()}))
    if stereo is None:
        meta['teacher_depth'] = 'camera'
        (d / 'meta.json').write_text(json.dumps(meta, indent=1))
    fb = meta['fx'] * meta['baseline']
    t0 = time.time()
    for i, f in enumerate(frames):
        if (d / f"{f['name']}_seg.png").exists():
            continue
        left = cv2.imread(str(d / f"{f['name']}_left.png"), cv2.IMREAD_GRAYSCALE)
        if stereo is None:
            depth = cv2.imread(str(d / f"{f['name']}_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        else:
            right = cv2.imread(str(d / f"{f['name']}_right.png"), cv2.IMREAD_GRAYSCALE)
            disp = run_stereo(stereo, left, right)
            depth = np.where(disp > 0.5, fb / np.maximum(disp, 1e-3), 0).astype(np.float32)
        np.save(d / f"{f['name']}_fsdepth.npy", depth.astype(np.float16))
        cv2.imwrite(str(d / f"{f['name']}_seg.png"), run_seg(proc, seg, left))
        if i % 20 == 0:
            print(f"{i}/{len(frames)} {(time.time() - t0) / (i + 1):.2f} s/frame", flush=True)


if __name__ == '__main__':
    main()

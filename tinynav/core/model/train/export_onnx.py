"""Export BevNet to ONNX (grid_sample needs opset >= 16), then build a TensorRT engine with e.g.
  trtexec --onnx=bevnet.onnx --fp16 --saveEngine=bevnet_fp16.plan

  python -m tinynav.core.model.train.export_onnx --ckpt runs/bev_v1/last.pt --out bevnet.onnx
"""
import argparse

import torch

from .bev_geometry import BEV_N, Z_SLICES
from .dataset import IMG_HW
from .model import BevNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    model = BevNet(pretrained=False).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    nz = len(Z_SLICES)
    inputs = (torch.rand(1, 1, *IMG_HW), torch.zeros(1, 4, BEV_N, BEV_N),
              torch.rand(1, nz * BEV_N, BEV_N, 2) * 2 - 1, torch.ones(1, nz, BEV_N, BEV_N))
    torch.onnx.export(model, inputs, args.out, opset_version=17, dynamo=False,
                      input_names=['img', 'raster', 'grid', 'valid'], output_names=['out'])
    print('wrote', args.out)


if __name__ == '__main__':
    main()

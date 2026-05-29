#!/usr/bin/env python3
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Merged model path")
    parser.add_argument("--out_onnx", required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_path)
    model.eval()

    dummy = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)
    os.makedirs(os.path.dirname(args.out_onnx) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.out_onnx,
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
    )
    print(f"ONNX exported to: {args.out_onnx}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import time

import cv2
import numpy as np

try:
    import tensorrt as trt
    from cuda import cudart
except ImportError as exc:
    raise SystemExit(
        "TensorRT and cuda-python are required. Run this inside the tinynav GPU container."
    ) from exc


class PIDNetTRT:
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(engine_path)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        _, self.stream = cudart.cudaStreamCreate()
        self._allocate_buffers()

    def _tensor_shape(self, name):
        shape = tuple(self.context.get_tensor_shape(name))
        if -1 not in shape:
            return shape
        try:
            _, opt_shape, _ = self.engine.get_tensor_profile_shape(name, 0)
            return tuple(int(v) for v in opt_shape)
        except Exception:
            return tuple(1 if v == -1 else int(v) for v in shape)

    def _allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self._tensor_shape(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype)
            _, device = cudart.cudaMalloc(host.nbytes)
            self.bindings.append(int(device))
            self.context.set_tensor_address(name, int(device))

            item = {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "host": host,
                "device": device,
                "nbytes": host.nbytes,
            }
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
                self.inputs.append(item)
            else:
                self.outputs.append(item)

        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected one PIDNet input, got {len(self.inputs)}")
        if len(self.outputs) != 1:
            raise RuntimeError(f"Expected one PIDNet output, got {len(self.outputs)}")

    @property
    def input_hw(self):
        shape = self.inputs[0]["shape"]
        if len(shape) != 4:
            raise RuntimeError(f"Expected NCHW/NHWC input, got {shape}")
        if shape[1] in (1, 3):
            return int(shape[2]), int(shape[3])
        return int(shape[1]), int(shape[2])

    def preprocess(self, image):
        h_net, w_net = self.input_hw
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w_net, h_net), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

        shape = self.inputs[0]["shape"]
        if shape[1] in (1, 3):
            image = np.transpose(image, (2, 0, 1))[None, ...]
        else:
            image = image[None, ...]
        return image.astype(self.inputs[0]["dtype"], copy=False)

    def infer(self, image):
        tensor = self.preprocess(image)
        np.copyto(self.inputs[0]["host"], tensor)
        cudart.cudaMemcpyAsync(
            self.inputs[0]["device"],
            self.inputs[0]["host"].ctypes.data,
            self.inputs[0]["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )
        self.context.execute_async_v3(stream_handle=self.stream)
        out = self.outputs[0]
        cudart.cudaMemcpyAsync(
            out["host"].ctypes.data,
            out["device"],
            out["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        cudart.cudaStreamSynchronize(self.stream)
        return out["host"].copy()


def _parse_channels(channels):
    if isinstance(channels, str):
        return [int(v.strip()) for v in channels.split(",") if v.strip()]
    if isinstance(channels, int):
        return [channels]
    return [int(v) for v in channels]


def floor_probability(raw_output, floor_channels=(0, 1)):
    out = np.asarray(raw_output)
    while out.ndim > 3 and out.shape[0] == 1:
        out = out[0]

    if out.ndim == 3 and out.shape[0] <= 256:
        logits = out
    elif out.ndim == 3 and out.shape[-1] <= 256:
        logits = np.transpose(out, (2, 0, 1))
    elif out.ndim == 2:
        return np.clip(out.astype(np.float32), 0.0, 1.0)
    else:
        raise RuntimeError(f"Unsupported PIDNet output shape: {raw_output.shape}")

    logits = logits.astype(np.float32)
    if logits.shape[0] == 1:
        return 1.0 / (1.0 + np.exp(-logits[0]))

    logits -= np.max(logits, axis=0, keepdims=True)
    prob = np.exp(logits)
    prob /= np.sum(prob, axis=0, keepdims=True)
    selected = _parse_channels(floor_channels)
    return np.sum(prob[selected], axis=0)


def make_overlay(image, prob, alpha=0.45, threshold=0.55):
    prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    floor = prob >= threshold
    color = np.zeros_like(image)
    color[floor] = (0, 180, 0)
    color[~floor] = (0, 0, 220)
    return cv2.addWeighted(color, alpha, image, 1.0 - alpha, 0.0), prob


def main():
    parser = argparse.ArgumentParser(description="Run a PIDNet TensorRT floor segmentation engine on one image.")
    parser.add_argument("--engine", required=True, help="Path to pidnet *.plan engine.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out", default="/tmp/pidnet_floor_overlay.png", help="Overlay output path.")
    parser.add_argument("--mask-out", default=None, help="Optional mono8 floor probability output path.")
    parser.add_argument(
        "--floor-channels",
        default="0,1",
        help="Comma-separated floor class channels for multi-class output. Cityscapes: 0=road, 1=sidewalk.",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Floor probability threshold for overlay.")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(args.image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3]

    runner = PIDNetTRT(args.engine)
    start = time.perf_counter()
    raw = runner.infer(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    prob = floor_probability(raw, args.floor_channels)
    overlay, prob_full = make_overlay(image_bgr, prob, threshold=args.threshold)
    cv2.imwrite(args.out, overlay)
    if args.mask_out:
        cv2.imwrite(args.mask_out, np.clip(prob_full * 255.0, 0, 255).astype(np.uint8))
    print(f"input={image.shape} engine_input={runner.inputs[0]['shape']} output={raw.shape} elapsed_ms={elapsed_ms:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

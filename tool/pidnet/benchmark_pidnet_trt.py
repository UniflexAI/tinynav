#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np

from pidnet_trt import PIDNetTRT


def main():
    parser = argparse.ArgumentParser(description="Benchmark a PIDNet TensorRT engine.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--image", default=None, help="Optional real image. Uses random image if omitted.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    runner = PIDNetTRT(args.engine)
    h, w = runner.input_hw
    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(args.image)
    else:
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        runner.infer(image)

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        runner.infer(image)
        times.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(times, dtype=np.float32)
    print(f"engine_input={runner.inputs[0]['shape']} output={runner.outputs[0]['shape']}")
    print(
        f"runs={args.runs} mean_ms={arr.mean():.2f} "
        f"p50_ms={np.percentile(arr, 50):.2f} p95_ms={np.percentile(arr, 95):.2f}"
    )


if __name__ == "__main__":
    main()

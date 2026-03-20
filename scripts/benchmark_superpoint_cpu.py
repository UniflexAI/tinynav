#!/usr/bin/env python3
import argparse
import os
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark SuperPoint ONNX on CPU with single-thread onnxruntime.")
    parser.add_argument(
        "--model",
        default="tinynav/models/superpoint_240x424_fp16.onnx",
        help="Path to the SuperPoint ONNX model.",
    )
    parser.add_argument("--height", type=int, default=240, help="Input height.")
    parser.add_argument("--width", type=int, default=424, help="Input width.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=200, help="Benchmark iterations.")
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is not installed in this environment.", file=sys.stderr)
        print("Install it in the devcontainer, then rerun this script.", file=sys.stderr)
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed in this environment.", file=sys.stderr)
        print("Install it in the devcontainer, then rerun this script.", file=sys.stderr)
        return 1

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 1

    model_size = os.path.getsize(args.model)
    if model_size < 1024:
        print(
            f"ERROR: model file looks too small ({model_size} bytes). It may still be a Git LFS pointer.",
            file=sys.stderr,
        )
        print("Run `git lfs pull` in the repo or ensure the model is available inside the devcontainer.", file=sys.stderr)
        return 1

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        args.model,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    input_meta = sess.get_inputs()[0]
    input_shape = [1, 1, args.height, args.width]
    x = np.random.randn(*input_shape).astype(np.float32)

    print(f"model: {args.model}")
    print(f"provider: {sess.get_providers()[0]}")
    print(f"input_name: {input_meta.name}")
    print(f"input_shape: {tuple(input_shape)}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print("threads: intra_op=1, inter_op=1")

    for _ in range(args.warmup):
        sess.run(None, {input_meta.name: x})

    times_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, {input_meta.name: x})
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.asarray(times_ms, dtype=np.float64)
    avg_ms = float(times_ms.mean())
    p50_ms = float(np.percentile(times_ms, 50))
    p95_ms = float(np.percentile(times_ms, 95))
    fps = 1000.0 / avg_ms

    print(f"avg_latency_ms: {avg_ms:.3f}")
    print(f"p50_latency_ms: {p50_ms:.3f}")
    print(f"p95_latency_ms: {p95_ms:.3f}")
    print(f"fps: {fps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

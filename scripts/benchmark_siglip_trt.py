#!/usr/bin/env python3
import argparse
import asyncio
import time

import cv2
import numpy as np

from tinynav.core.models_trt import SigLIPTRT
from tinynav.core.semantic_retrieval import normalize_embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SigLIP TensorRT image/text embedding")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="a hallway")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def summarize(name: str, values_s: list[float]):
    values_ms = np.asarray(values_s, dtype=np.float64) * 1000.0
    print(
        f"{name}: mean={values_ms.mean():.2f} ms, "
        f"p50={np.percentile(values_ms, 50):.2f} ms, "
        f"p95={np.percentile(values_ms, 95):.2f} ms, "
        f"n={len(values_ms)}"
    )


def main():
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    embedder = SigLIPTRT()
    image_latencies = []
    text_latencies = []
    image_embedding = None
    text_embedding = None

    for idx in range(args.iters + args.warmup):
        start = time.perf_counter()
        image_embedding = normalize_embedding(asyncio.run(embedder.encode_image(image)))
        elapsed = time.perf_counter() - start
        if idx >= args.warmup:
            image_latencies.append(elapsed)

    for idx in range(args.iters + args.warmup):
        start = time.perf_counter()
        text_embedding = normalize_embedding(asyncio.run(embedder.encode_text(args.text)))
        elapsed = time.perf_counter() - start
        if idx >= args.warmup:
            text_latencies.append(elapsed)

    summarize("SigLIP image embedding", image_latencies)
    summarize("SigLIP text embedding", text_latencies)
    print(f"image/text similarity: {float(image_embedding @ text_embedding):.4f}")


if __name__ == "__main__":
    main()

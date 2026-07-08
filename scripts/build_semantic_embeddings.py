#!/usr/bin/env python3
import argparse
import asyncio
import time

import numpy as np
from tqdm import tqdm

from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.models_trt import SigLIPTRT
from tinynav.core.semantic_retrieval import normalize_embedding


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill SigLIP semantic embeddings for an existing TinyNav map")
    parser.add_argument("--map_path", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Maximum keyframes to process; 0 means all")
    parser.add_argument("--warmup", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    db = TinyNavDB(args.map_path, is_scratch=False)
    embedder = SigLIPTRT()
    timestamps = sorted(db.depths.keys())
    if args.limit > 0:
        timestamps = timestamps[: args.limit]

    latencies = []
    try:
        for index, timestamp in enumerate(tqdm(timestamps, desc="semantic embeddings")):
            _depth, _embedding, _features, rgb_loader, infra1_loader = db.get_depth_embedding_features_images(timestamp)
            image = rgb_loader()
            if image is None:
                image = infra1_loader()
            if image is None:
                raise RuntimeError(f"No RGB or infra1 image found for timestamp {timestamp}")

            start = time.perf_counter()
            semantic_embedding = normalize_embedding(asyncio.run(embedder.encode_image(image)))
            elapsed = time.perf_counter() - start
            db.set_semantic_embedding(timestamp, semantic_embedding)
            if index >= args.warmup:
                latencies.append(elapsed)

        if latencies:
            latency_ms = np.asarray(latencies, dtype=np.float64) * 1000.0
            print(
                "SigLIP image embedding latency after warmup: "
                f"mean={latency_ms.mean():.2f} ms, "
                f"p50={np.percentile(latency_ms, 50):.2f} ms, "
                f"p95={np.percentile(latency_ms, 95):.2f} ms, "
                f"n={len(latency_ms)}"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()

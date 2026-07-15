#!/usr/bin/env python3
"""Generate VLAD descriptors + vocabulary for an existing tinynav map.

Usage:
    python tool/generate_vlad_map.py --map-path /path/to/map [--vocab-size 32] [--iterations 200]

This tool:
  1. Opens an existing map (TinyNavDB).
  2. For each keyframe, extracts DINOv2 patch tokens via TensorRT.
  3. Trains a K-means vocabulary on all patch tokens.
  4. Computes a VLAD descriptor per keyframe.
  5. Saves vlad_vocab.npy, vlad_descriptors.npy, vlad_timestamps.npy into the map directory.

Requirements:
  - The map must already have poses.npy and infra1 images stored in TinyNavDB.
  - DINOv2 TRT engine must exist at /tinynav/tinynav/models/dinov2_base_224x224_fp16_<arch>.plan
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

import numpy as np

# Ensure project root is on sys.path when running from repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.models_trt import Dinov2TRT
from tinynav.core.vlad import train_vocabulary, compute_vlad_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_vlad_map")


def main():
    parser = argparse.ArgumentParser(description="Generate VLAD descriptors for an existing map")
    parser.add_argument("--map-path", required=True, help="Path to the existing map directory")
    parser.add_argument("--vocab-size", type=int, default=32, help="Number of VLAD clusters (K)")
    parser.add_argument("--iterations", type=int, default=200, help="K-means iterations")
    parser.add_argument("--batch-size", type=int, default=1024, help="K-means minibatch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for k-means")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing VLAD files")
    args = parser.parse_args()

    map_path = args.map_path
    if not os.path.exists(os.path.join(map_path, "poses.npy")):
        logger.error(f"poses.npy not found in {map_path}")
        sys.exit(1)

    # Check for existing VLAD files.
    vlad_files = ["vlad_vocab.npy", "vlad_descriptors.npy", "vlad_timestamps.npy"]
    existing = [f for f in vlad_files if os.path.exists(os.path.join(map_path, f))]
    if existing and not args.overwrite:
        logger.warning(
            f"VLAD files already exist: {existing}. Use --overwrite to regenerate."
        )
        sys.exit(0)

    # Load poses to get timestamps.
    poses = np.load(os.path.join(map_path, "poses.npy"), allow_pickle=True).item()
    timestamps = sorted(poses.keys())
    logger.info(f"Map has {len(timestamps)} keyframes")

    # Open DB for reading.
    db = TinyNavDB(map_path, is_scratch=False)
    dinov2 = Dinov2TRT()
    logger.info("DINOv2 TRT engine loaded")

    # Step 1: Extract patch tokens for each keyframe.
    t0 = time.perf_counter()
    patch_tokens_list: list[np.ndarray] = []
    for i, ts in enumerate(timestamps):
        _, _, _, _, infra1_loader = db.get_depth_embedding_features_images(ts)
        image = infra1_loader() if infra1_loader is not None else None
        if image is None:
            logger.warning(f"Missing infra1 image for timestamp {ts}, skipping")
            patch_tokens_list.append(np.zeros((0, 768), dtype=np.float32))
            continue
        tokens = asyncio.run(dinov2.infer_patch_tokens(image))
        patch_tokens_list.append(tokens)
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Patch tokens extracted {i + 1}/{len(timestamps)} "
                f"({elapsed:.1f}s, {(i + 1) / elapsed:.1f} img/s)"
            )
    elapsed_extract = time.perf_counter() - t0
    logger.info(
        f"Patch token extraction done: {elapsed_extract:.1f}s "
        f"({len(timestamps) / elapsed_extract:.1f} img/s)"
    )

    # Step 2: Train vocabulary.
    valid_tokens = [t for t in patch_tokens_list if t.shape[0] > 0]
    if not valid_tokens:
        logger.error("No valid patch tokens extracted, cannot train VLAD")
        db.close()
        sys.exit(1)

    all_tokens = np.concatenate(valid_tokens, axis=0)
    logger.info(
        f"Training vocabulary: {all_tokens.shape[0]} tokens, "
        f"K={args.vocab_size}, dim={all_tokens.shape[1]}"
    )
    t1 = time.perf_counter()
    centres = train_vocabulary(
        all_tokens,
        vocab_size=args.vocab_size,
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    elapsed_train = time.perf_counter() - t1
    logger.info(f"Vocabulary trained: {centres.shape} ({elapsed_train:.1f}s)")

    # Step 3: Compute VLAD descriptors.
    t2 = time.perf_counter()
    vlad_descriptors = compute_vlad_batch(patch_tokens_list, centres)
    elapsed_vlad = time.perf_counter() - t2
    logger.info(
        f"VLAD descriptors computed: {vlad_descriptors.shape} ({elapsed_vlad:.1f}s)"
    )

    # Step 4: Save.
    np.save(os.path.join(map_path, "vlad_vocab.npy"), centres)
    np.save(os.path.join(map_path, "vlad_descriptors.npy"), vlad_descriptors)
    np.save(
        os.path.join(map_path, "vlad_timestamps.npy"),
        np.array(timestamps, dtype=np.int64),
    )
    logger.info(
        f"VLAD files saved to {map_path}: "
        f"vlad_vocab.npy {centres.shape}, "
        f"vlad_descriptors.npy {vlad_descriptors.shape}, "
        f"vlad_timestamps.npy ({len(timestamps)},)"
    )

    db.close()
    total = time.perf_counter() - t0
    logger.info(f"Done. Total: {total:.1f}s")


if __name__ == "__main__":
    main()

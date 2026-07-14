#!/usr/bin/env python3
import argparse
import os

import numpy as np

from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.superpoint_bow import SUPERPOINT_BOW_INDEX_FILENAME, SuperPointBoWRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SuperPoint BoW retrieval index for an existing TinyNav map.")
    parser.add_argument("map_path", help="Path to a built TinyNav map directory")
    parser.add_argument("--output", default=None, help=f"Output index path. Defaults to MAP/{SUPERPOINT_BOW_INDEX_FILENAME}")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--sample-limit", type=int, default=120000)
    parser.add_argument("--kmeans-iterations", type=int, default=30)
    args = parser.parse_args()

    map_path = os.path.abspath(args.map_path)
    output_path = args.output or os.path.join(map_path, SUPERPOINT_BOW_INDEX_FILENAME)
    poses_path = os.path.join(map_path, "poses.npy")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Map poses not found: {poses_path}")

    poses = np.load(poses_path, allow_pickle=True).item()
    timestamps = [int(timestamp) for timestamp in poses.keys()]
    db = TinyNavDB(map_path, is_scratch=False)
    retriever = SuperPointBoWRetriever(
        vocab_size=args.vocab_size,
        sample_limit=args.sample_limit,
        kmeans_iterations=args.kmeans_iterations,
    )
    try:
        retriever.build_from_feature_loader(timestamps, lambda timestamp: db.features[timestamp])
        retriever.save(output_path)
    finally:
        db.close()

    print(
        f"Saved SuperPoint BoW index to {output_path} "
        f"({len(retriever.timestamps)} keyframes, {len(retriever.vocab)} words)"
    )


if __name__ == "__main__":
    main()

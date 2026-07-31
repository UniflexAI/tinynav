from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from tinynav.core.bow_retrieval import BowConfig, build_bow_index
from tinynav.core.build_map_node import TinyNavDB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SuperPoint BoW retrieval index for an existing TinyNav map. "
            "This writes bow_index.npz so the standard map_node.py can use BoW retrieval."
        )
    )
    parser.add_argument("--tinynav_map_path", required=True, help="Path to an existing TinyNav map directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Defaults to <tinynav_map_path>/bow_index.npz",
    )
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--max_desc_per_image_for_train", type=int, default=40)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--kmeans_max_iter", type=int, default=40)
    parser.add_argument("--kmeans_eps", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing BoW index")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    map_path = Path(args.tinynav_map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"map path does not exist: {map_path}")
    if not map_path.is_dir():
        raise NotADirectoryError(f"map path is not a directory: {map_path}")

    poses_path = map_path / "poses.npy"
    features_path = map_path / "features.db"
    if not poses_path.exists():
        raise FileNotFoundError(f"missing map poses: {poses_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"missing SuperPoint features DB: {features_path}")

    output_path = Path(args.output) if args.output is not None else map_path / "bow_index.npz"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"BoW index already exists: {output_path}. Use --overwrite to rebuild it.")

    poses = np.load(poses_path, allow_pickle=True).item()
    timestamps = sorted(int(timestamp) for timestamp in poses.keys())
    if not timestamps:
        raise ValueError(f"map has no poses: {poses_path}")

    config = BowConfig(
        vocab_size=args.vocab_size,
        max_desc_per_image_for_train=args.max_desc_per_image_for_train,
        random_seed=args.random_seed,
        kmeans_max_iter=args.kmeans_max_iter,
        kmeans_eps=args.kmeans_eps,
    )

    logging.info("Building SuperPoint BoW index")
    logging.info("map_path=%s", map_path)
    logging.info("output_path=%s", output_path)
    logging.info("keyframes=%d", len(timestamps))
    logging.info("config=%s", config)

    db = TinyNavDB(str(map_path), is_scratch=False)
    try:
        build_bow_index(db, timestamps, output_path, config)
    finally:
        db.close()

    logging.info("Done. Wrote %s", output_path)
    logging.info("BoW index is ready; this map can now be used by map_node.py.")


if __name__ == "__main__":
    main()

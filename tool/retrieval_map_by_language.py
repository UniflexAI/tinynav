#!/usr/bin/env python3
import argparse
import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np

from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.models_trt import SigLIPTRT
from tinynav.core.semantic_retrieval import load_semantic_embedding_matrix, rank_semantic_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve the top matching TinyNav keyframe by text")
    parser.add_argument("--tinynav_map_path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default=None, help="Output image path. Defaults to /tmp/tinynav_retrieval_<timestamp>.jpg")
    parser.add_argument("--top_k", type=int, default=1)
    return parser.parse_args()


def load_map_timestamps(map_path: Path) -> list[int]:
    poses_path = map_path / "poses.npy"
    if not poses_path.exists():
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    poses = np.load(poses_path, allow_pickle=True).item()
    return [int(timestamp) for timestamp in poses.keys()]


def save_result_image(db: TinyNavDB, timestamp: int, output: str | None) -> Path:
    _depth, _embedding, _features, rgb_loader, infra1_loader = db.get_depth_embedding_features_images(timestamp)
    image = rgb_loader()
    if image is None:
        image = infra1_loader()
    if image is None:
        raise RuntimeError(f"No RGB or infra1 image found for timestamp {timestamp}")

    output_path = Path(output) if output is not None else Path(f"/tmp/tinynav_retrieval_{timestamp}.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    map_path = Path(args.tinynav_map_path)
    db = TinyNavDB(str(map_path), is_scratch=False)
    try:
        timestamps = load_map_timestamps(map_path)
        semantic_embeddings, semantic_timestamps = load_semantic_embedding_matrix(db, timestamps)
        if semantic_embeddings.shape[0] == 0:
            raise RuntimeError(f"No semantic embeddings found in {map_path}")

        embedder = SigLIPTRT()
        text_embedding = asyncio.run(embedder.encode_text(args.text))
        ranked = rank_semantic_embeddings(text_embedding, semantic_embeddings, semantic_timestamps, top_k=args.top_k)
        if not ranked:
            raise RuntimeError("No retrieval results")

        top_timestamp, top_score = ranked[0]
        output_path = save_result_image(db, top_timestamp, args.output)
        print(f"text={args.text}")
        print(f"timestamp={top_timestamp}")
        print(f"score={top_score:.6f}")
        print(f"image={output_path}")
        for rank, (timestamp, score) in enumerate(ranked, start=1):
            print(f"rank={rank} timestamp={timestamp} score={score:.6f}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

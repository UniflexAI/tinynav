#!/usr/bin/env python3
"""Copy a TinyNav map with a given set of keyframes physically removed.

Intended to run on the timestamp list a human confirmed after reviewing
generate_review_page.py's output (flagged by find_confusing_keyframes.py, then
manually vetted). Never mutates the source map -- always writes a full copy to
--output_path.

Only poses.npy and the per-keyframe shelve stores (features/depths/vlad_descriptors/
embeddings/semantic_embeddings/patch_tokens) are edited. map_node.py and
build_map_node.py derive the keyframe set entirely from poses.npy's keys, so this is
sufficient for the pruned keyframes to actually stop being used for relocalization.

infra1_images_db/rgb_images_db (video stores) are left untouched: nothing reads a
keyframe's image unless its timestamp is still in poses.npy, so the orphaned frames are
just harmless dead weight in the video file. Re-encoding the videos to drop them would
be slower and riskier (frame index / ts_to_idx must stay exactly in sync) for no
functional benefit.
"""
from __future__ import annotations

import argparse
import json
import shelve
import shutil
from pathlib import Path

import numpy as np

SHELVE_STORES = ["features", "embeddings", "semantic_embeddings", "vlad_descriptors", "patch_tokens", "depths"]


def _parse_exclude(args: argparse.Namespace) -> set[int]:
    values: set[int] = set()
    if args.exclude_timestamps:
        values.update(int(v.strip()) for v in args.exclude_timestamps.split(",") if v.strip())
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    values.add(int(line))
    if not values:
        raise ValueError("No timestamps to exclude: pass --exclude_timestamps and/or --exclude_file")
    return values


def prune_map(map_path: Path, output_path: Path, exclude: set[int]) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")
    if not (map_path / "poses.npy").exists():
        raise FileNotFoundError(f"Not a map directory (missing poses.npy): {map_path}")

    print(f"copying {map_path} -> {output_path} ...")
    shutil.copytree(map_path, output_path)

    poses_path = output_path / "poses.npy"
    poses = np.load(poses_path, allow_pickle=True).item()
    before = len(poses)
    removed_from_poses = sorted(ts for ts in exclude if int(ts) in poses)
    for ts in removed_from_poses:
        del poses[int(ts)]
    np.save(poses_path, poses, allow_pickle=True)

    removed_per_store: dict[str, int] = {}
    for name in SHELVE_STORES:
        db_path = output_path / f"{name}.db"
        if not db_path.exists():
            continue
        db = shelve.open(str(output_path / name))
        removed = 0
        try:
            for ts in exclude:
                key = str(int(ts))
                if key in db:
                    del db[key]
                    removed += 1
        finally:
            db.close()
        removed_per_store[name] = removed

    not_found = sorted(exclude - set(removed_from_poses))
    summary = {
        "map_path": str(map_path),
        "output_path": str(output_path),
        "keyframes_before": before,
        "keyframes_after": len(poses),
        "excluded_requested": len(exclude),
        "excluded_removed_from_poses": len(removed_from_poses),
        "excluded_not_found_in_poses": not_found,
        "removed_per_shelve_store": removed_per_store,
    }
    with (output_path / "prune_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_path", required=True, help="source map directory (read-only, never modified)")
    parser.add_argument("--output_path", required=True, help="new map directory to create (must not already exist)")
    parser.add_argument("--exclude_timestamps", default="", help="comma-separated keyframe timestamps (ns) to remove")
    parser.add_argument("--exclude_file", default="", help="optional: file with one timestamp per line, merged with --exclude_timestamps")
    args = parser.parse_args()

    exclude = _parse_exclude(args)
    summary = prune_map(Path(args.map_path), Path(args.output_path), exclude)

    print(f"keyframes: {summary['keyframes_before']} -> {summary['keyframes_after']}")
    print(f"removed {summary['excluded_removed_from_poses']}/{summary['excluded_requested']} requested timestamps")
    if summary["excluded_not_found_in_poses"]:
        print(f"warning: {len(summary['excluded_not_found_in_poses'])} requested timestamps were not in poses.npy (already absent?)")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()

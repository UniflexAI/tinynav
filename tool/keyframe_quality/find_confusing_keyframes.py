#!/usr/bin/env python3
"""Find map keyframes that repeatedly show up in geometrically-inconsistent retrieval results.

Map A (built from bag1 via build_map_node.py) is queried with every frame from an independent
evaluation bag (bag2). For each query, the top-K retrieval candidates' *known map positions*
are checked for spatial agreement: if the candidates don't cluster around one place, the
retrieval was almost certainly misled by perceptual aliasing (visually similar but physically
different locations) rather than a genuine revisit. Every map keyframe that took part in such a
"dispersed" retrieval gets a strike; keyframes that strike a lot relative to how often they're
retrieved at all are flagged as non-discriminative ("confusing") and worth pruning from the map.

This does not require ground truth for the query frames -- it only uses the map's own poses
(already known from mapping) and self-consistency of the candidate cluster.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

from tinynav.core.build_map_node import TinyNavDB, find_loop
from tinynav.core.models_trt import Dinov2TRT
from tinynav.core.vlad import compute_vlad


def iter_infra1_images(bag_path: str, topic: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topics:
        raise ValueError(f"Topic not found in bag: {topic}")
    msg_type = get_message(topics[topic])
    bridge = CvBridge()
    while reader.has_next():
        tpc, raw, ts_ns = reader.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(raw, msg_type)
        img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        yield int(ts_ns), img


def largest_cluster_fraction(positions: np.ndarray, radius_m: float) -> tuple[float, np.ndarray]:
    """Greedy radius-based clustering: fraction of points belonging to the densest cluster.

    Robust to a single stray outlier the way stdev / max-pairwise-distance aren't: 9-agree-1-off
    still reports 0.9, not "totally dispersed".
    """
    n = len(positions)
    if n <= 1:
        return 1.0, np.ones(n, dtype=bool)
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    neighbor_counts = (dists <= radius_m).sum(axis=1)
    center_idx = int(np.argmax(neighbor_counts))
    in_cluster = dists[center_idx] <= radius_m
    return float(in_cluster.sum()) / n, in_cluster


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    map_path = Path(args.map_path)
    map_poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    map_timestamps = sorted(int(t) for t in map_poses.keys())

    db = TinyNavDB(str(map_path), is_scratch=False)
    vlad_centres = db.metadata["vlad_centres"]
    map_vlad_descriptors = np.stack([db.vlad_descriptors[t] for t in map_timestamps]).astype(np.float32)
    embed_model = Dinov2TRT()

    keyframe_total: dict[int, int] = defaultdict(int)
    keyframe_bad: dict[int, int] = defaultdict(int)
    per_query_rows: list[dict[str, Any]] = []

    frame_idx = 0
    saved = 0
    for ts_ns, infra1 in iter_infra1_images(args.eval_bag_path, args.topic):
        if frame_idx % max(1, args.every_n) != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        if args.max_frames > 0 and saved >= args.max_frames:
            break

        patch_tokens = asyncio.run(embed_model.infer_patch_tokens(infra1))
        query_vec = compute_vlad(patch_tokens, vlad_centres)
        hits = find_loop(query_vec, map_vlad_descriptors, -1.0, args.topk)
        hits = [(int(map_timestamps[idx]), float(sim)) for idx, sim in hits]

        hits = [(ts, sim) for ts, sim in hits if sim >= args.min_similarity]
        saved += 1
        if len(hits) < 2:
            continue

        positions = np.array([map_poses[ts][:3, 3] for ts, _ in hits])
        cluster_fraction, _in_cluster = largest_cluster_fraction(positions, args.cluster_radius_m)
        dispersion = 1.0 - cluster_fraction
        is_bad = dispersion > args.dispersion_threshold

        for ts, sim in hits:
            keyframe_total[ts] += 1
            if is_bad:
                keyframe_bad[ts] += 1

        per_query_rows.append(
            {
                "query_timestamp_ns": int(ts_ns),
                "candidates": [{"timestamp_ns": ts, "similarity": sim} for ts, sim in hits],
                "cluster_fraction": cluster_fraction,
                "dispersion": dispersion,
                "is_bad": bool(is_bad),
            }
        )
        if saved % 20 == 0:
            print(f"processed={saved}")

    db.close()

    keyframe_report = []
    for ts, total in keyframe_total.items():
        if total < args.min_participation:
            continue
        bad = keyframe_bad.get(ts, 0)
        keyframe_report.append(
            {
                "timestamp_ns": ts,
                "total_participation": total,
                "bad_participation": bad,
                "badness_ratio": bad / total,
            }
        )
    keyframe_report.sort(key=lambda r: (r["badness_ratio"], r["total_participation"]), reverse=True)

    n_bad_queries = sum(1 for r in per_query_rows if r["is_bad"])
    summary = {
        "map_path": str(map_path),
        "eval_bag_path": args.eval_bag_path,
        "topk": args.topk,
        "min_similarity": args.min_similarity,
        "cluster_radius_m": args.cluster_radius_m,
        "dispersion_threshold": args.dispersion_threshold,
        "min_participation": args.min_participation,
        "query_count": len(per_query_rows),
        "bad_query_count": n_bad_queries,
        "bad_query_ratio": n_bad_queries / max(1, len(per_query_rows)),
        "flagged_keyframe_count": len(keyframe_report),
        "flagged_keyframes": keyframe_report,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    if args.out_per_query_jsonl:
        per_query_path = Path(args.out_per_query_jsonl)
        per_query_path.parent.mkdir(parents=True, exist_ok=True)
        with per_query_path.open("w", encoding="utf-8") as f:
            for row in per_query_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_path", required=True, help="pre-built TinyNav map directory (built from bag1)")
    parser.add_argument("--eval_bag_path", required=True, help="bag2: independent probe bag, read directly (raw frames)")
    parser.add_argument("--topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--topk", type=int, default=5, help="candidates per query (default 5: more tolerant than production's top-3, since this is an offline optimization tool)")
    parser.add_argument("--min_similarity", type=float, default=-1.0, help="drop candidates below this similarity before dispersion check")
    parser.add_argument("--cluster_radius_m", type=float, default=1.0, help="candidates within this radius of each other count as agreeing")
    parser.add_argument("--dispersion_threshold", type=float, default=0.4, help="retrieval is 'bad' if less than a majority of candidates cluster together (default: reject if largest cluster < 60%% of topk)")
    parser.add_argument("--min_participation", type=int, default=3, help="ignore map keyframes retrieved fewer than this many times (avoid noise from rarely-hit keyframes)")
    parser.add_argument("--every_n", type=int, default=5, help="subsample eval bag frames")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = no cap")
    parser.add_argument("--out_json", default="tinynav_temp/confusing_keyframes.json")
    parser.add_argument("--out_per_query_jsonl", default="", help="optional: dump every query's candidates+dispersion verdict")
    args = parser.parse_args()

    summary = run_eval(args)

    print(f"\nqueries={summary['query_count']}  bad_queries={summary['bad_query_count']} ({summary['bad_query_ratio']:.1%})")
    print(f"flagged {summary['flagged_keyframe_count']} keyframes (participation >= {args.min_participation})")
    for r in summary["flagged_keyframes"][:15]:
        print(f"  ts={r['timestamp_ns']}  badness={r['badness_ratio']:.2f}  ({r['bad_participation']}/{r['total_participation']})")
    print(f"\nfull report: {args.out_json}")


if __name__ == "__main__":
    main()

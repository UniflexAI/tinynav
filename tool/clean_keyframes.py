#!/usr/bin/env python3
"""Clean a TinyNav map: detect and remove non-discriminative ("confusing") keyframes.

Map A (built via build_map_node.py) is queried with every frame from an independent probe
bag. For each query, the top-K retrieval candidates' *known map positions* are checked for
spatial agreement: if the candidates don't cluster around one place, the retrieval was almost
certainly misled by perceptual aliasing (visually similar but physically different locations)
rather than a genuine revisit. Every map keyframe that took part in such a "dispersed"
retrieval gets a strike; any keyframe with >= 1 strike (and enough retrievals overall to be
meaningful, --min_participation) is non-discriminative and gets removed.

Writes two things, never touching the source map:
  - a pruned copy of the map (--output_map_path)
  - a folder with an HTML report + evidence images explaining what was removed and why
    (--output_report_dir), showing each removed keyframe's worst query/candidate evidence
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shelve
import shutil
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

SHELVE_STORES = ["features", "embeddings", "semantic_embeddings", "vlad_descriptors", "patch_tokens", "depths"]


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


def detect(args: argparse.Namespace) -> dict[str, Any]:
    """Query the map with the probe bag; return per-keyframe stats and per-query evidence rows."""
    map_path = Path(args.map_path)
    map_poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    map_timestamps = sorted(int(t) for t in map_poses.keys())

    db = TinyNavDB(str(map_path), is_scratch=False)
    vlad_centres = db.metadata["vlad_centres"]
    map_vlad_descriptors = np.stack([db.vlad_descriptors[t] for t in map_timestamps]).astype(np.float32)
    db.close()

    embed_model = Dinov2TRT()
    keyframe_total: dict[int, int] = defaultdict(int)
    keyframe_outlier_count: dict[int, int] = defaultdict(int)
    per_query_rows: list[dict[str, Any]] = []

    frame_idx = 0
    saved = 0
    for ts_ns, infra1 in iter_infra1_images(args.bag_path, args.topic):
        if frame_idx % max(1, args.every_n) != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        if args.max_frames > 0 and saved >= args.max_frames:
            break
        saved += 1

        patch_tokens = asyncio.run(embed_model.infer_patch_tokens(infra1))
        query_vec = compute_vlad(patch_tokens, vlad_centres)
        hits = find_loop(query_vec, map_vlad_descriptors, -1.0, args.topk)
        hits = [(int(map_timestamps[idx]), float(sim)) for idx, sim in hits]
        hits = [(ts, sim) for ts, sim in hits if sim >= args.min_similarity]
        if len(hits) < 2:
            continue

        positions = np.array([map_poses[ts][:3, 3] for ts, _ in hits])
        cluster_fraction, in_cluster_mask = largest_cluster_fraction(positions, args.cluster_radius_m)
        dispersion = 1.0 - cluster_fraction
        is_dispersed = dispersion > args.dispersion_threshold

        # Only the outlier candidates (outside the agreeing majority) are actually responsible
        # for the perceptual aliasing -- a candidate that's part of the consensus cluster is a
        # legitimate match and shouldn't be struck just because other candidates in the same
        # top-K were dispersed.
        for (ts, _sim), in_cluster in zip(hits, in_cluster_mask):
            keyframe_total[ts] += 1
            if is_dispersed and not in_cluster:
                keyframe_outlier_count[ts] += 1

        per_query_rows.append(
            {
                "query_timestamp_ns": int(ts_ns),
                "candidates": [
                    {"timestamp_ns": ts, "similarity": sim, "in_cluster": bool(in_cluster)}
                    for (ts, sim), in_cluster in zip(hits, in_cluster_mask)
                ],
                "dispersion": dispersion,
                "is_dispersed": bool(is_dispersed),
            }
        )
        if saved % 20 == 0:
            print(f"processed={saved}")

    keyframe_stats = []
    for ts, total in keyframe_total.items():
        if total < args.min_participation:
            continue
        outlier_count = keyframe_outlier_count.get(ts, 0)
        keyframe_stats.append(
            {
                "timestamp_ns": ts,
                "total_retrievals": total,
                "outlier_count": outlier_count,
                "outlier_ratio": outlier_count / total,
            }
        )
    keyframe_stats.sort(key=lambda r: (r["outlier_ratio"], r["total_retrievals"]), reverse=True)

    n_dispersed_queries = sum(1 for r in per_query_rows if r["is_dispersed"])
    return {
        "keyframe_stats": keyframe_stats,
        "per_query_rows": per_query_rows,
        "query_count": len(per_query_rows),
        "dispersed_query_count": n_dispersed_queries,
    }


def prune_map(map_path: Path, output_path: Path, exclude: set[int]) -> dict:
    """Copy map_path to output_path with `exclude` timestamps removed. Source is never modified.

    Only poses.npy and the per-keyframe shelve stores are edited. map_node.py and
    build_map_node.py derive the keyframe set entirely from poses.npy's keys, so this is
    sufficient for the pruned keyframes to actually stop being used for relocalization.
    Video stores are left untouched: nothing reads a keyframe's image once its timestamp is
    gone from poses.npy, so the orphaned frames are just harmless dead weight in the video file.
    """
    if output_path.exists():
        raise FileExistsError(f"Output path already exists: {output_path}")
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

    return {
        "keyframes_before": before,
        "keyframes_after": len(poses),
        "removed_per_shelve_store": removed_per_store,
    }


def _select_examples(per_query_rows: list[dict], removed_ts: set[int], max_per_keyframe: int) -> dict[int, list[dict]]:
    """For each removed keyframe, pick its worst-dispersion bad queries where it was actually
    the struck outlier (in_cluster=False) -- these are the rows that explain the strike, not
    just any bad query it happened to co-occur in as an innocent (in-cluster) bystander."""
    examples: dict[int, list[dict]] = {ts: [] for ts in removed_ts}
    for row in per_query_rows:
        if not row["is_dispersed"]:
            continue
        outlier_ts = {c["timestamp_ns"] for c in row["candidates"] if not c["in_cluster"]}
        for ts in outlier_ts & removed_ts:
            examples[ts].append(row)
    for ts, rows in examples.items():
        rows.sort(key=lambda r: r["dispersion"], reverse=True)
        examples[ts] = rows[:max_per_keyframe]
    return examples


def _save_thumb(img: np.ndarray | None, path: Path, width: int, quality: int) -> bool:
    if img is None:
        return False
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))))
    return bool(cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality]))


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8">
<title>Clean Keyframes Report</title>
<style>
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 24px; background: #fafafa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) { body { background: #17181c; color: #e8e8e8; } }
h1 { font-size: 20px; margin: 0 0 4px; }
.summary { font-size: 13px; opacity: 0.75; margin-bottom: 20px; line-height: 1.6; }
.card { border: 1px solid rgba(128,128,128,0.3); border-radius: 10px; margin-bottom: 10px; background: rgba(128,128,128,0.04); }
.card-head { display: flex; align-items: center; gap: 12px; padding: 10px 14px; cursor: pointer; }
.card-head img.thumb-sm { width: 72px; border-radius: 6px; }
.card-head .meta { flex: 1; font-size: 13px; }
.detail { display: none; padding: 0 14px 14px; border-top: 1px solid rgba(128,128,128,0.2); }
.example { margin-top: 12px; }
.example .row { display: flex; gap: 8px; flex-wrap: nowrap; align-items: flex-start; margin-top: 6px; }
.example .candidates-row { display: flex; gap: 8px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px; min-width: 0; flex: 1 1 auto; }
.thumb-box { text-align: center; font-size: 10px; flex: 0 0 auto; }
.thumb-box img { width: 110px; border-radius: 5px; display: block; }
.thumb-box.in-cluster img { outline: 3px solid #2d9a4e; }
.thumb-box.out-cluster img { outline: 3px solid #d2452d; }
.thumb-box.is-self img { box-shadow: 0 0 0 2px gold; }
.arrow { align-self: center; font-size: 18px; opacity: 0.5; flex: 0 0 auto; }
</style>
</head>
<body>
<h1>Clean Keyframes Report</h1>
<div class="summary" id="summary"></div>
<div id="cards"></div>
<script>
const DATA = __DATA_JSON__;

function fmtPct(x) { return (x * 100).toFixed(0) + "%"; }

document.getElementById("summary").innerHTML =
  `map: <b>${DATA.map_path}</b> &nbsp;|&nbsp; bag: <b>${DATA.bag_path}</b><br>` +
  `queries=${DATA.query_count}, of which ${DATA.dispersed_query_count} (${fmtPct(DATA.dispersed_query_count / Math.max(1, DATA.query_count))}) had spatially-dispersed top-K candidates<br>` +
  `topk=${DATA.topk} cluster_radius_m=${DATA.cluster_radius_m} dispersion_threshold=${DATA.dispersion_threshold} ` +
  `min_participation=${DATA.min_participation}<br>` +
  `removed <b>${DATA.removed_keyframe_count}</b> keyframes`;

function candBox(c) {
  const cls = (c.in_cluster ? "in-cluster" : "out-cluster") + (c.is_self ? " is-self" : "");
  return `<div class="thumb-box ${cls}">
    <img src="${c.thumb}">
    ts=${c.timestamp_ns}<br>sim=${c.similarity.toFixed(3)}${c.is_self ? " (this kf)" : ""}
  </div>`;
}

function exampleBlock(ex) {
  const cands = ex.candidates.map(candBox).join("");
  const nOut = ex.candidates.filter(c => !c.in_cluster).length;
  return `<div class="example">
    <div style="font-size:12px;opacity:0.8">
      one query (ts=${ex.query_timestamp_ns}) where this keyframe was pulled in as a candidate:
      ${nOut}/${ex.candidates.length} of the top-K candidates <u>for this single query</u> didn't spatially agree
      (dispersion=${ex.dispersion.toFixed(2)}). This is one instance out of the keyframe's overall track record below.
    </div>
    <div class="row">
      <div class="thumb-box"><img src="${ex.query_thumb}">query</div>
      <div class="arrow">&rarr;</div>
      <div class="candidates-row">${cands}</div>
    </div>
  </div>`;
}

document.getElementById("cards").innerHTML = DATA.records.map((r, i) => {
  const examples = r.examples.map(exampleBlock).join("") || "<i>no example retrievals captured</i>";
  return `<div class="card">
    <div class="card-head" onclick="toggleDetail(${i})">
      <img class="thumb-sm" src="${r.thumb}">
      <div class="meta">ts=${r.timestamp_ns} &nbsp; track record across the whole bag: retrieved as a candidate
        <b>${r.total_retrievals}</b> times total (not capped at topk -- this sums every query in the bag where
        it showed up), and was the spatial outlier in <b>${r.outlier_count}</b> of them
        (outlier_ratio=${(r.outlier_ratio * 100).toFixed(0)}%) &rarr; removed for non-discriminative matching</div>
    </div>
    <div class="detail" id="detail-${i}">${examples}</div>
  </div>`;
}).join("");

function toggleDetail(i) {
  const el = document.getElementById(`detail-${i}`);
  el.style.display = el.style.display === "block" ? "none" : "block";
}
</script>
</body></html>
"""


def write_report(
    args: argparse.Namespace,
    map_path: Path,
    detection: dict[str, Any],
    removed_ts: set[int],
    output_report_dir: Path,
) -> None:
    if output_report_dir.exists():
        shutil.rmtree(output_report_dir)
    images_dir = output_report_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    keyframe_stats = detection["keyframe_stats"]
    examples_by_ts = _select_examples(detection["per_query_rows"], removed_ts, args.max_examples_per_keyframe)

    needed_map_ts: set[int] = set(removed_ts)
    needed_query_ts: set[int] = set()
    for rows in examples_by_ts.values():
        for row in rows:
            needed_query_ts.add(int(row["query_timestamp_ns"]))
            needed_map_ts.update(int(c["timestamp_ns"]) for c in row["candidates"])

    print(f"extracting {len(needed_map_ts)} map thumbnails, {len(needed_query_ts)} query thumbnails...")
    db = TinyNavDB(str(map_path), is_scratch=False)
    map_thumbs: dict[int, str] = {}
    try:
        for ts in needed_map_ts:
            img = db.infra1_video_db.read(ts)
            if img is None:
                img = db.rgb_video_db.read(ts)
            rel_path = f"images/kf_{ts}.jpg"
            if _save_thumb(img, output_report_dir / rel_path, args.thumbnail_width, args.jpeg_quality):
                map_thumbs[ts] = rel_path
    finally:
        db.close()

    query_thumbs: dict[int, str] = {}
    if needed_query_ts:
        for ts, img in iter_infra1_images(args.bag_path, args.topic):
            if ts in needed_query_ts and ts not in query_thumbs:
                rel_path = f"images/query_{ts}.jpg"
                if _save_thumb(img, output_report_dir / rel_path, args.thumbnail_width, args.jpeg_quality):
                    query_thumbs[ts] = rel_path
                if len(query_thumbs) == len(needed_query_ts):
                    break

    kf_by_ts = {kf["timestamp_ns"]: kf for kf in keyframe_stats}
    records = []
    for ts in sorted(removed_ts):
        kf = kf_by_ts[ts]
        example_payload = []
        for row in examples_by_ts.get(ts, []):
            cand_payload = [
                {
                    "timestamp_ns": int(c["timestamp_ns"]),
                    "similarity": c["similarity"],
                    "in_cluster": c["in_cluster"],
                    "is_self": int(c["timestamp_ns"]) == ts,
                    "thumb": map_thumbs.get(int(c["timestamp_ns"]), ""),
                }
                for c in row["candidates"]
            ]
            example_payload.append(
                {
                    "query_timestamp_ns": int(row["query_timestamp_ns"]),
                    "query_thumb": query_thumbs.get(int(row["query_timestamp_ns"]), ""),
                    "dispersion": row["dispersion"],
                    "candidates": cand_payload,
                }
            )
        records.append(
            {
                "timestamp_ns": ts,
                "total_retrievals": kf["total_retrievals"],
                "outlier_count": kf["outlier_count"],
                "outlier_ratio": kf["outlier_ratio"],
                "thumb": map_thumbs.get(ts, ""),
                "examples": example_payload,
            }
        )

    report_data = {
        "map_path": str(map_path),
        "bag_path": args.bag_path,
        "topk": args.topk,
        "cluster_radius_m": args.cluster_radius_m,
        "dispersion_threshold": args.dispersion_threshold,
        "min_participation": args.min_participation,
        "query_count": detection["query_count"],
        "dispersed_query_count": detection["dispersed_query_count"],
        "removed_keyframe_count": len(removed_ts),
        "records": records,
    }
    (output_report_dir / "report.json").write_text(json.dumps(report_data, ensure_ascii=True, indent=2), encoding="utf-8")
    html = _PAGE_TEMPLATE.replace("__DATA_JSON__", json.dumps(report_data, ensure_ascii=True))
    (output_report_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_path", required=True, help="pre-built TinyNav map directory (read-only, never modified)")
    parser.add_argument("--bag_path", required=True, help="independent probe bag, read directly (raw frames)")
    parser.add_argument("--output_map_path", default="", help="new pruned map directory (must not exist). Default: <map_path>_cleaned")
    parser.add_argument("--output_report_dir", default="", help="folder for index.html + images/. Default: <output_map_path>_report")
    parser.add_argument("--topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--topk", type=int, default=5, help="candidates per query")
    parser.add_argument("--min_similarity", type=float, default=-1.0, help="drop candidates below this similarity before dispersion check")
    parser.add_argument("--cluster_radius_m", type=float, default=1.0, help="candidates within this radius of each other count as agreeing")
    parser.add_argument("--dispersion_threshold", type=float, default=0.4, help="a retrieval is 'dispersed' if less than a majority of candidates cluster together")
    parser.add_argument("--min_participation", type=int, default=3, help="ignore map keyframes retrieved fewer than this many times")
    parser.add_argument("--every_n", type=int, default=5, help="subsample probe bag frames")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = no cap")
    parser.add_argument("--max_examples_per_keyframe", type=int, default=4, help="worst-dispersion example retrievals shown per removed keyframe in the report")
    parser.add_argument("--thumbnail_width", type=int, default=200)
    parser.add_argument("--jpeg_quality", type=int, default=70)
    args = parser.parse_args()

    map_path = Path(args.map_path)
    output_map_path = Path(args.output_map_path) if args.output_map_path else map_path.parent / f"{map_path.name}_cleaned"
    output_report_dir = Path(args.output_report_dir) if args.output_report_dir else Path(f"{output_map_path}_report")

    print(f"detecting confusing keyframes: map={map_path} bag={args.bag_path}")
    detection = detect(args)
    keyframe_stats = detection["keyframe_stats"]
    removed_ts = {kf["timestamp_ns"] for kf in keyframe_stats if kf["outlier_count"] > 0}

    print(f"\nqueries={detection['query_count']}  dispersed_queries={detection['dispersed_query_count']}")
    print(f"{len(keyframe_stats)} keyframes had >= {args.min_participation} retrievals; "
          f"{len(removed_ts)} were the spatial outlier at least once and will be removed")

    prune_summary = prune_map(map_path, output_map_path, removed_ts)
    print(f"keyframes: {prune_summary['keyframes_before']} -> {prune_summary['keyframes_after']}")
    print(f"wrote pruned map to {output_map_path}")

    write_report(args, map_path, detection, removed_ts, output_report_dir)
    print(f"wrote report to {output_report_dir / 'index.html'}")


if __name__ == "__main__":
    main()

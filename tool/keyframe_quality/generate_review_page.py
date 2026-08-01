#!/usr/bin/env python3
"""Build a self-contained HTML review page for keyframes find_confusing_keyframes.py flagged.

Consumes that script's --out_json (flagged keyframe list) and --out_per_query_jsonl (every
query's candidates + verdict) plus the same map used for detection. For each flagged keyframe,
pulls a handful of its worst ("bad") retrievals -- the query frame and every candidate in that
top-K set -- as thumbnails, so a human can eyeball whether the flag is real (genuinely
generic-looking place) or a false positive, before deciding what to prune.

Every flagged keyframe starts checked ("will be removed"); unchecking one in the page means
"keep it despite the flag". The page never touches the map itself -- it only renders a
prune_map.py command line for the keyframes still checked once you're done reviewing.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from find_confusing_keyframes import iter_infra1_images, largest_cluster_fraction
from tinynav.core.build_map_node import TinyNavDB


def _encode_thumb(img: np.ndarray | None, width: int, quality: int) -> str:
    if img is None:
        return ""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(h * width / w))))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _select_examples(per_query_path: Path, flagged_ts: set[int], max_per_keyframe: int) -> dict[int, list[dict]]:
    examples: dict[int, list[dict]] = {ts: [] for ts in flagged_ts}
    with per_query_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row["is_bad"]:
                continue
            candidate_ts = {c["timestamp_ns"] for c in row["candidates"]}
            for ts in candidate_ts & flagged_ts:
                examples[ts].append(row)
    for ts, rows in examples.items():
        rows.sort(key=lambda r: r["dispersion"], reverse=True)
        examples[ts] = rows[:max_per_keyframe]
    return examples


def build_records(args: argparse.Namespace) -> dict[str, Any]:
    map_path = Path(args.map_path)
    with Path(args.flagged_json).open(encoding="utf-8") as f:
        summary = json.load(f)
    # find_confusing_keyframes.py's --out_json lists every keyframe meeting --min_participation,
    # sorted by badness_ratio -- most have badness_ratio == 0 (fine, just there so the sorted cutoff
    # is visible). Only keyframes that actually took part in >=1 bad retrieval belong in a review
    # page whose checkboxes default to "will be removed".
    flagged_keyframes = [kf for kf in summary["flagged_keyframes"] if kf["bad_participation"] > 0]
    flagged_ts = {int(kf["timestamp_ns"]) for kf in flagged_keyframes}

    if not summary.get("self_query", False) and not args.eval_bag_path:
        raise ValueError("--eval_bag_path is required: flagged_json was not generated with --self_query")

    examples_by_ts = _select_examples(Path(args.per_query_jsonl), flagged_ts, args.max_examples_per_keyframe)

    map_poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    map_poses = {int(ts): np.asarray(pose) for ts, pose in map_poses.items()}
    cluster_radius_m = float(summary["cluster_radius_m"])
    self_query = bool(summary.get("self_query", False))

    needed_map_ts: set[int] = set(flagged_ts)
    needed_query_ts: set[int] = set()
    for rows in examples_by_ts.values():
        for row in rows:
            needed_query_ts.add(int(row["query_timestamp_ns"]))
            needed_map_ts.update(int(c["timestamp_ns"]) for c in row["candidates"])
    if self_query:
        # queries *are* map keyframes here, so their thumbnails come from the same map video DB.
        needed_map_ts.update(needed_query_ts)

    print(f"extracting {len(needed_map_ts)} map thumbnails" + ("" if self_query else f", {len(needed_query_ts)} query thumbnails") + "...")

    db = TinyNavDB(str(map_path), is_scratch=False)
    map_thumbs: dict[int, str] = {}
    try:
        for ts in needed_map_ts:
            img = db.infra1_video_db.read(ts)
            if img is None:
                img = db.rgb_video_db.read(ts)
            map_thumbs[ts] = _encode_thumb(img, args.thumbnail_width, args.jpeg_quality)
    finally:
        db.close()

    if self_query:
        query_thumbs = map_thumbs
    else:
        query_thumbs = {}
        if needed_query_ts:
            for ts, img in iter_infra1_images(args.eval_bag_path, args.topic):
                if ts in needed_query_ts and ts not in query_thumbs:
                    query_thumbs[ts] = _encode_thumb(img, args.thumbnail_width, args.jpeg_quality)
                    if len(query_thumbs) == len(needed_query_ts):
                        break

    records = []
    for kf in flagged_keyframes:
        ts = int(kf["timestamp_ns"])
        example_payload = []
        for row in examples_by_ts.get(ts, []):
            candidates = row["candidates"]
            positions = np.array([map_poses[int(c["timestamp_ns"])][:3, 3] for c in candidates])
            _, in_cluster_mask = largest_cluster_fraction(positions, cluster_radius_m)
            cand_payload = []
            for c, in_cluster in zip(candidates, in_cluster_mask):
                c_ts = int(c["timestamp_ns"])
                cand_payload.append(
                    {
                        "timestamp_ns": c_ts,
                        "similarity": c["similarity"],
                        "in_cluster": bool(in_cluster),
                        "is_self": c_ts == ts,
                        "thumb": map_thumbs.get(c_ts, ""),
                    }
                )
            example_payload.append(
                {
                    "query_timestamp_ns": int(row["query_timestamp_ns"]),
                    "query_thumb": query_thumbs.get(int(row["query_timestamp_ns"]), ""),
                    "dispersion": row["dispersion"],
                    "cluster_fraction": row["cluster_fraction"],
                    "candidates": cand_payload,
                }
            )
        records.append(
            {
                "timestamp_ns": ts,
                "total_participation": kf["total_participation"],
                "bad_participation": kf["bad_participation"],
                "badness_ratio": kf["badness_ratio"],
                "thumb": map_thumbs.get(ts, ""),
                "examples": example_payload,
            }
        )

    return {
        "map_path": summary["map_path"],
        "self_query": self_query,
        "eval_bag_path": summary["eval_bag_path"],
        "topk": summary["topk"],
        "cluster_radius_m": summary["cluster_radius_m"],
        "dispersion_threshold": summary["dispersion_threshold"],
        "min_participation": summary["min_participation"],
        "query_count": summary["query_count"],
        "bad_query_count": summary["bad_query_count"],
        "bad_query_ratio": summary["bad_query_ratio"],
        "flagged_keyframe_count": len(records),
        "flagged_keyframe_count_raw": summary["flagged_keyframe_count"],
        "records": records,
    }


_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh"><head>
<meta charset="utf-8">
<title>Confusing Keyframe Review</title>
<style>
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 24px; background: #fafafa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) { body { background: #17181c; color: #e8e8e8; } }
h1 { font-size: 20px; margin: 0 0 4px; }
.summary { font-size: 13px; opacity: 0.75; margin-bottom: 20px; line-height: 1.6; }
.controls { position: sticky; top: 0; background: inherit; padding: 12px 0; border-bottom: 1px solid rgba(128,128,128,0.3); margin-bottom: 16px; z-index: 10; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.controls input[type=text] { padding: 6px 8px; font-size: 13px; min-width: 220px; }
button { padding: 6px 14px; font-size: 13px; cursor: pointer; border-radius: 6px; border: 1px solid rgba(128,128,128,0.4); background: #2d7dd2; color: white; }
button.secondary { background: transparent; color: inherit; }
#cmd-box { display: none; margin-top: 10px; }
#cmd-box pre { background: rgba(128,128,128,0.12); padding: 12px; border-radius: 8px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
.card { border: 1px solid rgba(128,128,128,0.3); border-radius: 10px; margin-bottom: 10px; background: rgba(128,128,128,0.04); }
.card-head { display: flex; align-items: center; gap: 12px; padding: 10px 14px; cursor: pointer; }
.card-head img.thumb-sm { width: 72px; border-radius: 6px; }
.card-head .meta { flex: 1; font-size: 13px; }
.badness-bar { display: inline-block; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #d2452d, #d2452d); vertical-align: middle; }
.detail { display: none; padding: 0 14px 14px; border-top: 1px solid rgba(128,128,128,0.2); }
.example { margin-top: 12px; }
.example .row { display: flex; gap: 8px; flex-wrap: nowrap; align-items: flex-start; margin-top: 6px; }
.example .candidates-row { display: flex; gap: 8px; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px; min-width: 0; flex: 1 1 auto; }
.thumb-box { text-align: center; font-size: 10px; flex: 0 0 auto; }
.thumb-box img { width: 110px; border-radius: 5px; display: block; }
.thumb-box.in-cluster img { outline: 3px solid #2d9a4e; }
.thumb-box.out-cluster img { outline: 3px solid #d2452d; }
.thumb-box.is-self img { outline-offset: 2px; box-shadow: 0 0 0 2px gold; }
.arrow { align-self: center; font-size: 18px; opacity: 0.5; flex: 0 0 auto; }
label.chk { display: flex; align-items: center; gap: 6px; font-size: 12px; white-space: nowrap; }
</style>
</head>
<body>
<h1>Confusing Keyframe Review</h1>
<div class="summary" id="summary"></div>
<div class="controls">
  <label class="chk"><input type="checkbox" id="select-all" checked> 全选/全不选</label>
  <span style="opacity:0.6">输出文件夹:</span>
  <input type="text" id="output-name" placeholder="e.g. map_gt_pruned">
  <button id="gen-btn">生成裁剪命令</button>
  <button class="secondary" id="copy-btn" style="display:none">复制</button>
</div>
<div id="cmd-box"><pre id="cmd-text"></pre></div>
<div id="cards"></div>
<script>
const DATA = __DATA_JSON__;
const excluded = new Set(DATA.records.map(r => r.timestamp_ns));

function fmtPct(x) { return (x * 100).toFixed(0) + "%"; }

function renderSummary() {
  const s = DATA;
  const evalLabel = s.self_query ? "self-query (map queried against itself)" : s.eval_bag_path;
  document.getElementById("summary").innerHTML =
    `map: <b>${s.map_path}</b> &nbsp;|&nbsp; eval: <b>${evalLabel}</b><br>` +
    `queries=${s.query_count} bad_queries=${s.bad_query_count} (${fmtPct(s.bad_query_ratio)}) &nbsp;|&nbsp; ` +
    `topk=${s.topk} cluster_radius_m=${s.cluster_radius_m} dispersion_threshold=${s.dispersion_threshold} min_participation=${s.min_participation}<br>` +
    `flagged keyframes (bad_participation &gt; 0): <b>${s.flagged_keyframe_count}</b> out of ${s.flagged_keyframe_count_raw} with participation &gt;= min_participation`;
}

function candBox(c) {
  const cls = c.is_self ? "is-self" : (c.in_cluster ? "in-cluster" : "out-cluster");
  return `<div class="thumb-box ${cls}">
    <img src="${c.thumb}">
    ts=${c.timestamp_ns}<br>sim=${c.similarity.toFixed(3)}${c.is_self ? " (this kf)" : ""}
  </div>`;
}

function exampleBlock(ex) {
  const cands = ex.candidates.map(candBox).join("");
  return `<div class="example">
    <div style="font-size:12px;opacity:0.8">query ts=${ex.query_timestamp_ns} &mdash; dispersion=${ex.dispersion.toFixed(2)} (cluster_fraction=${ex.cluster_fraction.toFixed(2)})</div>
    <div class="row">
      <div class="thumb-box"><img src="${ex.query_thumb}">query</div>
      <div class="arrow">&rarr;</div>
      <div class="candidates-row">${cands}</div>
    </div>
  </div>`;
}

function renderCards() {
  const container = document.getElementById("cards");
  container.innerHTML = DATA.records.map((r, i) => {
    const examples = r.examples.map(exampleBlock).join("") || "<i>no per-query examples captured</i>";
    return `<div class="card">
      <div class="card-head" onclick="toggleDetail(${i})">
        <input type="checkbox" class="kf-check" data-ts="${r.timestamp_ns}" checked onclick="event.stopPropagation(); toggleExclude(${r.timestamp_ns}, this.checked)">
        <img class="thumb-sm" src="${r.thumb}">
        <div class="meta">
          ts=${r.timestamp_ns} &nbsp; badness=${(r.badness_ratio*100).toFixed(0)}% (${r.bad_participation}/${r.total_participation})
        </div>
      </div>
      <div class="detail" id="detail-${i}">${examples}</div>
    </div>`;
  }).join("");
}

function toggleDetail(i) {
  const el = document.getElementById(`detail-${i}`);
  el.style.display = el.style.display === "block" ? "none" : "block";
}

function toggleExclude(ts, checked) {
  if (checked) excluded.add(ts); else excluded.delete(ts);
}

document.getElementById("select-all").addEventListener("change", (e) => {
  document.querySelectorAll(".kf-check").forEach(cb => {
    cb.checked = e.target.checked;
    toggleExclude(parseInt(cb.dataset.ts, 10), e.target.checked);
  });
});

document.getElementById("gen-btn").addEventListener("click", () => {
  const outName = document.getElementById("output-name").value.trim() || "map_pruned";
  const tsList = Array.from(excluded).sort((a, b) => a - b).join(",");
  const cmd = `uv run python tool/keyframe_quality/prune_map.py \\\\\n` +
    `  --map_path ${DATA.map_path} \\\\\n` +
    `  --output_path tinynav_db/maps/${outName} \\\\\n` +
    `  --exclude_timestamps ${tsList}`;
  document.getElementById("cmd-text").textContent = cmd;
  document.getElementById("cmd-box").style.display = "block";
  document.getElementById("copy-btn").style.display = "inline-block";
});

document.getElementById("copy-btn").addEventListener("click", () => {
  const text = document.getElementById("cmd-text").textContent;
  navigator.clipboard.writeText(text).catch(() => {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  });
});

renderSummary();
renderCards();
</script>
</body></html>
"""


def write_page(data: dict[str, Any], out_html: Path) -> None:
    payload = json.dumps(data, ensure_ascii=True)
    page = _PAGE_TEMPLATE.replace("__DATA_JSON__", payload)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_path", required=True, help="same map passed to find_confusing_keyframes.py")
    parser.add_argument("--eval_bag_path", default="", help="same eval bag passed to find_confusing_keyframes.py. Not needed if that run used --self_query")
    parser.add_argument("--topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--flagged_json", required=True, help="find_confusing_keyframes.py --out_json output")
    parser.add_argument("--per_query_jsonl", required=True, help="find_confusing_keyframes.py --out_per_query_jsonl output")
    parser.add_argument("--max_examples_per_keyframe", type=int, default=4, help="cap example retrievals shown per flagged keyframe (worst dispersion first)")
    parser.add_argument("--thumbnail_width", type=int, default=200)
    parser.add_argument("--jpeg_quality", type=int, default=70)
    parser.add_argument("--out_html", default="tinynav_temp/confusing_keyframes_review.html")
    args = parser.parse_args()

    data = build_records(args)
    write_page(data, Path(args.out_html))
    print(f"wrote {args.out_html} ({len(data['records'])} flagged keyframes)")


if __name__ == "__main__":
    main()

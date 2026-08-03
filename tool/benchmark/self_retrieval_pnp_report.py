#!/usr/bin/env python3
import argparse
import asyncio
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.models_trt import LightGlueTRT


def load_pose_dict(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    return {int(k): v for k, v in data.items()}


def sample_last_fraction(timestamps: list[int], fraction: float, max_samples: int) -> list[int]:
    start = int(len(timestamps) * max(0.0, min(1.0, 1.0 - fraction)))
    selected = timestamps[start:]
    if max_samples > 0 and len(selected) > max_samples:
        idx = np.round(np.linspace(0, len(selected) - 1, max_samples)).astype(int)
        selected = [selected[i] for i in idx]
    return selected


def topk_history(query_desc: np.ndarray, timestamps: list[int], descriptors: np.ndarray, query_ts: int, top_k: int, gap_ns: int):
    valid = [i for i, ts in enumerate(timestamps) if ts + gap_ns < query_ts]
    if not valid:
        return []
    valid_desc = descriptors[valid]
    sims = valid_desc @ query_desc
    order = np.argsort(sims)[::-1][:top_k]
    return [(valid[int(i)], float(sims[int(i)])) for i in order]


def match_keypoints(matcher: LightGlueTRT, feats0: dict, feats1: dict, image_shape=np.array([848, 480], dtype=np.int64)):
    result = asyncio.run(
        matcher.infer(
            feats0["kpts"],
            feats1["kpts"],
            feats0["descps"],
            feats1["descps"],
            feats0["mask"],
            feats1["mask"],
            image_shape,
            image_shape,
        )
    )
    match_indices = result["match_indices"][0]
    valid_mask = match_indices != -1
    keypoints0 = feats0["kpts"][0][valid_mask]
    keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
    return np.asarray(keypoints0), np.asarray(keypoints1)


def keypoints_to_world(keypoints: np.ndarray, depth: np.ndarray, pose_camera_to_world: np.ndarray, K: np.ndarray):
    points_camera = []
    valid = []
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    height, width = depth.shape[:2]
    for kp in keypoints:
        u = int(round(float(kp[0])))
        v = int(round(float(kp[1])))
        z = float(depth[v, u]) if 0 <= u < width and 0 <= v < height else 0.0
        if 0.0 < z < 50.0:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_camera.append([x, y, z])
            valid.append(True)
        else:
            points_camera.append([0.0, 0.0, 0.0])
            valid.append(False)
    points_camera = np.asarray(points_camera, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    points_world = points_camera @ pose_camera_to_world[:3, :3].T + pose_camera_to_world[:3, 3]
    return points_world, valid


def pnp_pose(points_world: np.ndarray, points_2d: np.ndarray, K: np.ndarray, min_inliers: int):
    if len(points_2d) <= 4:
        return False, np.eye(4), np.empty((0,), dtype=np.int32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(points_world, points_2d, K, None)
    if not success or inliers is None or len(inliers) < min_inliers:
        return False, np.eye(4), np.empty((0,), dtype=np.int32)
    T_world_to_camera = np.eye(4)
    rot, _ = cv2.Rodrigues(rvec)
    T_world_to_camera[:3, :3] = rot
    T_world_to_camera[:3, 3] = tvec.reshape(3)
    return True, np.linalg.inv(T_world_to_camera), inliers.reshape(-1).astype(np.int32)


def to_bgr(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def inlier_distribution(query_kpts: np.ndarray, inlier_indices: np.ndarray, image_shape: tuple[int, int]):
    if len(inlier_indices) == 0:
        return {
            "median_y_norm": None,
            "lower_half_ratio": 0.0,
            "bottom_third_ratio": 0.0,
            "grid_coverage_4x4": 0,
        }
    h, w = image_shape[:2]
    pts = query_kpts[inlier_indices]
    x_norm = np.clip(pts[:, 0] / max(w, 1), 0.0, 1.0)
    y_norm = np.clip(pts[:, 1] / max(h, 1), 0.0, 1.0)
    xs = np.clip((x_norm * 4).astype(np.int32), 0, 3)
    ys = np.clip((y_norm * 4).astype(np.int32), 0, 3)
    return {
        "median_y_norm": float(np.median(y_norm)),
        "lower_half_ratio": float(np.mean(y_norm >= 0.5)),
        "bottom_third_ratio": float(np.mean(y_norm >= 2.0 / 3.0)),
        "grid_coverage_4x4": int(len(set(zip(xs.tolist(), ys.tolist())))),
    }


def draw_match_image(ref_image, query_image, ref_kpts, query_kpts, inlier_indices, max_lines=80):
    left = to_bgr(ref_image)
    right = to_bgr(query_image)
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    canvas = np.concatenate([left, right], axis=1)
    offset = left.shape[1]
    indices = list(map(int, inlier_indices))
    if len(indices) > max_lines:
        pick = np.round(np.linspace(0, len(indices) - 1, max_lines)).astype(int)
        indices = [indices[i] for i in pick]
    for idx in indices:
        p0 = tuple(np.round(ref_kpts[idx]).astype(int))
        p1 = tuple(np.round(query_kpts[idx]).astype(int) + np.array([offset, 0]))
        cv2.circle(canvas, p0, 3, (37, 99, 235), -1)
        cv2.circle(canvas, p1, 3, (249, 115, 22), -1)
        cv2.line(canvas, p0, p1, (34, 197, 94), 1, cv2.LINE_AA)
    return canvas


def img_uri(path: Path) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def fmt(value, precision=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def stat(rows, key):
    vals = np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=float)
    if len(vals) == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(np.max(vals)),
    }


def main():
    parser = argparse.ArgumentParser(description="Self-retrieval/PnP diagnostics for eval map keyframes.")
    parser.add_argument("--map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--last-fraction", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--history-gap-s", type=float, default=10.0)
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--image-limit", type=int, default=40)
    args = parser.parse_args()

    map_dir = Path(args.map)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    poses = load_pose_dict(map_dir / "poses.npy")
    timestamps = sorted(poses)
    sample_timestamps = sample_last_fraction(timestamps, args.last_fraction, args.max_samples)
    db = TinyNavDB(str(map_dir), is_scratch=False)
    matcher = LightGlueTRT()
    K = np.load(map_dir / "intrinsics.npy")
    descriptors = np.stack([db.vlad_descriptors[t] for t in timestamps])
    gap_ns = int(args.history_gap_s * 1e9)

    rows = []
    for sample_index, query_ts in enumerate(sample_timestamps):
        query_depth, _, query_features, _, query_image_loader = db.get_depth_embedding_features_images(query_ts)
        query_image = query_image_loader()
        query_desc = db.vlad_descriptors[query_ts]
        candidates = topk_history(query_desc, timestamps, descriptors, query_ts, args.top_k, gap_ns)

        candidate_rows = []
        best = None
        best_image = None
        for rank, (cand_idx, sim) in enumerate(candidates, start=1):
            cand_ts = timestamps[cand_idx]
            cand_depth, _, cand_features, _, cand_image_loader = db.get_depth_embedding_features_images(cand_ts)
            cand_image = cand_image_loader()
            ref_kpts, query_kpts = match_keypoints(matcher, cand_features, query_features)
            match_count = int(len(query_kpts))
            points_world, depth_valid = keypoints_to_world(ref_kpts, cand_depth, poses[cand_ts], K)
            valid_world = points_world[depth_valid].astype(np.float32)
            valid_query = query_kpts[depth_valid].astype(np.float32)
            success, pose_camera_to_world, pnp_inliers_valid = pnp_pose(valid_world, valid_query, K, args.min_inliers)
            original_valid_indices = np.where(depth_valid)[0]
            original_inliers = original_valid_indices[pnp_inliers_valid] if success else np.empty((0,), dtype=np.int32)
            dist_pose = float(np.linalg.norm(poses[query_ts][:3, 3] - poses[cand_ts][:3, 3]))
            dz_pose = float(poses[query_ts][2, 3] - poses[cand_ts][2, 3])
            pnp_error = None
            pnp_dz = None
            if success:
                delta = pose_camera_to_world[:3, 3] - poses[query_ts][:3, 3]
                pnp_error = float(np.linalg.norm(delta))
                pnp_dz = float(delta[2])
            dist = inlier_distribution(query_kpts, original_inliers, query_image.shape[:2])
            row = {
                "rank": rank,
                "candidate_timestamp_ns": int(cand_ts),
                "candidate_time_s": float(cand_ts) / 1e9,
                "similarity": float(sim),
                "time_gap_s": float((query_ts - cand_ts) / 1e9),
                "pose_distance_m": dist_pose,
                "pose_dz_query_minus_candidate_m": dz_pose,
                "match_count": match_count,
                "landmark_count": int(len(valid_query)),
                "pnp_success": bool(success),
                "pnp_inlier_count": int(len(original_inliers)),
                "pnp_inlier_ratio": float(len(original_inliers) / max(len(valid_query), 1)),
                "pnp_error_to_query_pose_m": pnp_error,
                "pnp_dz_to_query_pose_m": pnp_dz,
                **dist,
            }
            candidate_rows.append(row)
            if best is None or row["pnp_inlier_count"] > best["pnp_inlier_count"]:
                best = row
                best_image = draw_match_image(cand_image, query_image, ref_kpts, query_kpts, original_inliers)

        image_path = image_dir / f"sample_{sample_index:03d}_{query_ts}.jpg"
        if best_image is not None:
            cv2.imwrite(str(image_path), best_image)
        sample_row = {
            "sample_index": sample_index,
            "query_timestamp_ns": int(query_ts),
            "query_time_s": float(query_ts) / 1e9,
            "progress_in_map": float(timestamps.index(query_ts) / max(len(timestamps) - 1, 1)),
            "query_xyz": poses[query_ts][:3, 3].tolist(),
            "best": best,
            "candidates": candidate_rows,
            "image": image_path.name,
        }
        rows.append(sample_row)

    (output_dir / "self_retrieval_rows.json").write_text(json.dumps(rows, indent=2))
    write_html(output_dir, rows, args)
    db.close()
    print(f"report: {output_dir / 'index.html'}")


def summarize(rows):
    best_rows = [r["best"] for r in rows if r["best"] is not None]
    succ = [r for r in best_rows if r["pnp_success"]]
    return {
        "query_count": len(rows),
        "has_candidate_count": len(best_rows),
        "pnp_success_count": len(succ),
        "pnp_success_ratio": len(succ) / max(len(rows), 1),
        "best_rank_counts": {str(rank): sum(1 for r in succ if r["rank"] == rank) for rank in range(1, 6)},
        "similarity": stat(succ, "similarity"),
        "time_gap_s": stat(succ, "time_gap_s"),
        "pose_distance_m": stat(succ, "pose_distance_m"),
        "pose_dz_query_minus_candidate_m": stat(succ, "pose_dz_query_minus_candidate_m"),
        "pnp_inlier_count": stat(succ, "pnp_inlier_count"),
        "pnp_inlier_ratio": stat(succ, "pnp_inlier_ratio"),
        "pnp_error_to_query_pose_m": stat(succ, "pnp_error_to_query_pose_m"),
        "pnp_dz_to_query_pose_m": stat(succ, "pnp_dz_to_query_pose_m"),
        "inlier_lower_half_ratio": stat(succ, "lower_half_ratio"),
        "inlier_bottom_third_ratio": stat(succ, "bottom_third_ratio"),
        "inlier_grid_coverage_4x4": stat(succ, "grid_coverage_4x4"),
    }


def write_html(output_dir: Path, rows: list[dict], args):
    summary = summarize(rows)
    best_rows = [r for r in rows if r["best"] is not None]
    hard_rows = sorted(
        best_rows,
        key=lambda r: (
            not r["best"]["pnp_success"],
            r["best"].get("pnp_error_to_query_pose_m") or 0.0,
            r["best"].get("bottom_third_ratio") or 0.0,
        ),
        reverse=True,
    )[: args.image_limit]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{r['sample_index']}</td><td>{r['query_time_s']:.3f}</td><td>{r['progress_in_map']*100:.1f}%</td>"
        f"<td>{b['rank'] if b else 'n/a'}</td><td>{fmt(b['similarity'] if b else None)}</td>"
        f"<td>{fmt(b['time_gap_s'] if b else None)}</td><td>{fmt(b['pose_distance_m'] if b else None)}</td>"
        f"<td>{fmt(b['pose_dz_query_minus_candidate_m'] if b else None)}</td><td>{b['pnp_inlier_count'] if b else 0}</td>"
        f"<td>{fmt(b['pnp_inlier_ratio'] if b else None)}</td><td>{fmt(b['pnp_error_to_query_pose_m'] if b else None)}</td>"
        f"<td>{fmt(b['pnp_dz_to_query_pose_m'] if b else None)}</td><td>{fmt(b['lower_half_ratio'] if b else None)}</td>"
        f"<td>{fmt(b['bottom_third_ratio'] if b else None)}</td><td>{b['grid_coverage_4x4'] if b else 0}</td>"
        "</tr>"
        for r in rows
        for b in [r["best"]]
    )
    cards = []
    for r in hard_rows:
        b = r["best"]
        image_path = output_dir / "images" / r["image"]
        cand_rows = "\n".join(
            "<tr>"
            f"<td>{c['rank']}</td><td>{c['candidate_time_s']:.3f}</td><td>{c['similarity']:.4f}</td>"
            f"<td>{c['time_gap_s']:.1f}</td><td>{c['pose_distance_m']:.3f}</td><td>{c['pose_dz_query_minus_candidate_m']:.3f}</td>"
            f"<td>{c['match_count']}</td><td>{c['landmark_count']}</td><td>{c['pnp_inlier_count']}</td>"
            f"<td>{c['pnp_inlier_ratio']:.3f}</td><td>{fmt(c['pnp_error_to_query_pose_m'])}</td><td>{fmt(c['pnp_dz_to_query_pose_m'])}</td>"
            f"<td>{fmt(c['lower_half_ratio'])}</td><td>{fmt(c['bottom_third_ratio'])}</td><td>{c['grid_coverage_4x4']}</td>"
            "</tr>"
            for c in r["candidates"]
        )
        cards.append(
            f"""
<section>
<h2>sample {r['sample_index']} · query {r['query_time_s']:.3f}s · best rank {b['rank']}</h2>
<p>pnp err={fmt(b['pnp_error_to_query_pose_m'])}m, dz={fmt(b['pnp_dz_to_query_pose_m'])}m,
inliers={b['pnp_inlier_count']}, bottom-third={fmt(b['bottom_third_ratio'])}, lower-half={fmt(b['lower_half_ratio'])}.</p>
<img src="{img_uri(image_path)}" />
<table><tr><th>rank</th><th>cand time</th><th>sim</th><th>time gap</th><th>pose dist</th><th>pose dz</th><th>matches</th><th>landmarks</th><th>inliers</th><th>ratio</th><th>pnp err</th><th>pnp dz</th><th>lower 1/2</th><th>bottom 1/3</th><th>grid</th></tr>{cand_rows}</table>
</section>
"""
        )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TinyNav Eval Self Retrieval PnP Report</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:28px;background:#f8fafc;color:#111827}}
section{{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #e5e7eb;padding:6px;text-align:left}}th{{background:#eef2ff}}
img{{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}}pre{{background:#111827;color:#f8fafc;padding:12px;border-radius:8px;overflow:auto}}
</style></head><body>
<h1>TinyNav Eval Self Retrieval PnP Report</h1>
<p>Samples: eval map last {args.last_fraction*100:.0f}%, max {args.max_samples}; retrieval searches historical keyframes only, with {args.history_gap_s:.1f}s temporal gap. Match image: left=history candidate, right=query; green lines=PnP inliers.</p>
<section><h2>Summary</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre></section>
<section><h2>All Samples</h2><table><tr><th>#</th><th>query time</th><th>progress</th><th>best rank</th><th>sim</th><th>gap s</th><th>pose dist</th><th>pose dz</th><th>inliers</th><th>ratio</th><th>pnp err</th><th>pnp dz</th><th>lower 1/2</th><th>bottom 1/3</th><th>grid</th></tr>{table_rows}</table></section>
{''.join(cards)}
</body></html>"""
    (output_dir / "index.html").write_text(html_text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import asyncio
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from tinynav.core.build_map_node import TinyNavDB, find_loop
from tinynav.core.models_trt import LightGlueTRT


def rotation_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_err = rot_a.T @ rot_b
    value = np.clip((np.trace(rot_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def load_pose_dict(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    return {int(k): v for k, v in data.items()}


def sample_last_fraction(timestamps: list[int], fraction: float, max_samples: int) -> list[int]:
    if not timestamps:
        return []
    start = int(len(timestamps) * max(0.0, min(1.0, 1.0 - fraction)))
    selected = timestamps[start:]
    if max_samples > 0 and len(selected) > max_samples:
        idx = np.round(np.linspace(0, len(selected) - 1, max_samples)).astype(int)
        selected = [selected[i] for i in idx]
    return selected


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
        if 0 <= u < width and 0 <= v < height:
            z = float(depth[v, u])
        else:
            z = 0.0
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


def write_png(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def img_uri(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def fmt(value, precision=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Diagnose GT top-k retrieval and PnP for eval map keyframes.")
    parser.add_argument("--map-gt", required=True)
    parser.add_argument("--map-eval", required=True)
    parser.add_argument("--benchmark-report", required=True, help="Existing benchmark report directory containing metrics.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--last-fraction", type=float, default=0.20)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-inliers", type=int, default=50)
    parser.add_argument("--image-limit", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    map_gt = Path(args.map_gt)
    map_eval = Path(args.map_eval)
    gt_poses = load_pose_dict(map_gt / "poses.npy")
    eval_poses = load_pose_dict(map_eval / "poses.npy")
    T_eval_to_gt = np.asarray(json.loads((Path(args.benchmark_report) / "metrics.json").read_text())["transform_map_eval_to_gt"], dtype=float)

    gt_db = TinyNavDB(str(map_gt), is_scratch=False)
    eval_db = TinyNavDB(str(map_eval), is_scratch=False)
    matcher = LightGlueTRT()

    gt_timestamps = sorted(gt_poses)
    eval_timestamps = sorted(eval_poses)
    sample_timestamps = sample_last_fraction(eval_timestamps, args.last_fraction, args.max_samples)
    gt_descriptors = np.stack([gt_db.vlad_descriptors[t] for t in gt_timestamps])
    gt_K = np.load(map_gt / "intrinsics.npy")

    rows = []
    image_rows = []
    for sample_index, eval_ts in enumerate(sample_timestamps):
        eval_desc = eval_db.vlad_descriptors[eval_ts]
        candidates = list(reversed(find_loop(eval_desc, gt_descriptors, -1.0, args.top_k)))
        eval_depth, _, eval_features, _, eval_image_loader = eval_db.get_depth_embedding_features_images(eval_ts)
        eval_image = eval_image_loader()
        expected_pose = T_eval_to_gt @ eval_poses[eval_ts]

        cand_rows = []
        best = None
        best_image = None
        for rank, (gt_idx, sim) in enumerate(candidates, start=1):
            gt_ts = gt_timestamps[int(gt_idx)]
            gt_depth, _, gt_features, _, gt_image_loader = gt_db.get_depth_embedding_features_images(gt_ts)
            gt_image = gt_image_loader()
            ref_kpts_all, query_kpts_all = match_keypoints(matcher, gt_features, eval_features)
            match_count = int(len(query_kpts_all))
            points_world, depth_valid = keypoints_to_world(ref_kpts_all, gt_depth, gt_poses[gt_ts], gt_K)
            points_world_valid = points_world[depth_valid].astype(np.float32)
            query_valid = query_kpts_all[depth_valid].astype(np.float32)
            landmark_count = int(len(query_valid))
            success, pose_camera_to_world, pnp_inliers = pnp_pose(points_world_valid, query_valid, gt_K, args.min_inliers)
            inlier_count = int(len(pnp_inliers)) if success else 0
            inlier_ratio = float(inlier_count / max(landmark_count, 1))
            trans_error = None
            rot_error = None
            dz = None
            if success:
                delta = pose_camera_to_world[:3, 3] - expected_pose[:3, 3]
                trans_error = float(np.linalg.norm(delta))
                dz = float(delta[2])
                rot_error = rotation_error_deg(pose_camera_to_world[:3, :3], expected_pose[:3, :3])
            row = {
                "rank": rank,
                "gt_timestamp_ns": int(gt_ts),
                "similarity": float(sim),
                "match_count": match_count,
                "landmark_count": landmark_count,
                "pnp_success": bool(success),
                "pnp_inlier_count": inlier_count,
                "pnp_inlier_ratio": inlier_ratio,
                "translation_error_m": trans_error,
                "rotation_error_deg": rot_error,
                "dz_m": dz,
            }
            cand_rows.append(row)
            if best is None or inlier_count > best["pnp_inlier_count"]:
                best = row
                if success:
                    # PnP inlier indices are in depth-valid arrays; map back to original matched arrays.
                    valid_indices = np.where(depth_valid)[0]
                    original_inliers = valid_indices[pnp_inliers]
                    best_image = draw_match_image(gt_image, eval_image, ref_kpts_all, query_kpts_all, original_inliers)
                else:
                    best_image = np.concatenate([to_bgr(gt_image), to_bgr(eval_image)], axis=1)

        best = best or {}
        image_path = image_dir / f"sample_{sample_index:03d}_{eval_ts}.jpg"
        if best_image is not None:
            write_png(image_path, best_image)
        sample_row = {
            "sample_index": sample_index,
            "eval_timestamp_ns": int(eval_ts),
            "time_s": float(eval_ts) / 1e9,
            "progress_in_eval_map": float(eval_timestamps.index(eval_ts) / max(len(eval_timestamps) - 1, 1)),
            "expected_eval_in_gt_xyz": expected_pose[:3, 3].tolist(),
            "best_rank": best.get("rank"),
            "best_similarity": best.get("similarity"),
            "best_match_count": best.get("match_count", 0),
            "best_landmark_count": best.get("landmark_count", 0),
            "best_pnp_success": best.get("pnp_success", False),
            "best_pnp_inlier_count": best.get("pnp_inlier_count", 0),
            "best_pnp_inlier_ratio": best.get("pnp_inlier_ratio", 0.0),
            "best_translation_error_m": best.get("translation_error_m"),
            "best_rotation_error_deg": best.get("rotation_error_deg"),
            "best_dz_m": best.get("dz_m"),
            "candidates": cand_rows,
            "image": str(image_path.name),
        }
        rows.append(sample_row)

    (output_dir / "topk_pnp_rows.json").write_text(json.dumps(rows, indent=2))
    write_html(output_dir, rows, args)
    gt_db.close()
    eval_db.close()
    print(f"report: {output_dir / 'index.html'}")


def summarize(rows):
    success = [r for r in rows if r["best_pnp_success"]]
    def stat(key):
        vals = np.asarray([r[key] for r in success if r[key] is not None], dtype=float)
        if len(vals) == 0:
            return {"mean": None, "median": None, "p90": None, "max": None}
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "max": float(np.max(vals)),
        }
    abs_dz_vals = np.asarray(
        [abs(r["best_dz_m"]) for r in success if r["best_dz_m"] is not None],
        dtype=float,
    )
    if len(abs_dz_vals) == 0:
        abs_dz = {"mean": None, "median": None, "p90": None, "max": None}
    else:
        abs_dz = {
            "mean": float(np.mean(abs_dz_vals)),
            "median": float(np.median(abs_dz_vals)),
            "p90": float(np.percentile(abs_dz_vals, 90)),
            "max": float(np.max(abs_dz_vals)),
        }
    return {
        "count": len(rows),
        "success_count": len(success),
        "success_ratio": len(success) / max(len(rows), 1),
        "best_rank_counts": {str(rank): sum(1 for r in success if r["best_rank"] == rank) for rank in range(1, 6)},
        "inliers": stat("best_pnp_inlier_count"),
        "inlier_ratio": stat("best_pnp_inlier_ratio"),
        "translation_error_m": stat("best_translation_error_m"),
        "abs_dz_m": abs_dz,
    }


def write_html(output_dir: Path, rows: list[dict], args):
    summary = summarize(rows)
    by_error = sorted(
        [r for r in rows if r["best_translation_error_m"] is not None],
        key=lambda r: r["best_translation_error_m"],
        reverse=True,
    )
    by_fail = [r for r in rows if not r["best_pnp_success"]]
    image_rows = (by_fail + by_error)[: args.image_limit]

    table_rows = "\n".join(
        "<tr>"
        f"<td>{r['sample_index']}</td><td>{r['time_s']:.3f}</td><td>{r['progress_in_eval_map']*100:.1f}%</td>"
        f"<td>{r['best_rank']}</td><td>{fmt(r['best_similarity'])}</td><td>{r['best_match_count']}</td>"
        f"<td>{r['best_landmark_count']}</td><td>{r['best_pnp_inlier_count']}</td><td>{fmt(r['best_pnp_inlier_ratio'])}</td>"
        f"<td>{fmt(r['best_translation_error_m'])}</td><td>{fmt(r['best_dz_m'])}</td>"
        "</tr>"
        for r in rows
    )
    image_cards = []
    for r in image_rows:
        image_path = output_dir / "images" / r["image"]
        uri = img_uri(image_path) if image_path.exists() else ""
        cand_rows = "\n".join(
            "<tr>"
            f"<td>{c['rank']}</td><td>{c['gt_timestamp_ns']}</td><td>{c['similarity']:.4f}</td>"
            f"<td>{c['match_count']}</td><td>{c['landmark_count']}</td><td>{c['pnp_inlier_count']}</td>"
            f"<td>{c['pnp_inlier_ratio']:.3f}</td><td>{fmt(c['translation_error_m'])}</td><td>{fmt(c['dz_m'])}</td>"
            "</tr>"
            for c in r["candidates"]
        )
        image_cards.append(
            f"""
            <section>
              <h2>sample {r['sample_index']} · t={r['time_s']:.3f}s · best rank {r['best_rank']}</h2>
              <p>best trans err={fmt(r['best_translation_error_m'])}m, dz={fmt(r['best_dz_m'])}m, inliers={r['best_pnp_inlier_count']}, ratio={fmt(r['best_pnp_inlier_ratio'])}</p>
              <img src="{uri}" />
              <table><tr><th>rank</th><th>gt timestamp</th><th>sim</th><th>matches</th><th>landmarks</th><th>pnp inliers</th><th>ratio</th><th>trans err</th><th>dz</th></tr>{cand_rows}</table>
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TinyNav TopK PnP Relocalization Diagnostics</title>
<style>
body{{font-family:Inter,Arial,sans-serif;margin:28px;background:#f8fafc;color:#111827}}
section{{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:18px;margin:18px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e7eb;padding:7px;text-align:left}}th{{background:#eef2ff}}
img{{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}}pre{{background:#111827;color:#f8fafc;padding:12px;border-radius:8px;overflow:auto}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.metric{{background:#eef2ff;padding:12px;border-radius:8px}}
</style></head><body>
<h1>TinyNav TopK PnP Relocalization Diagnostics</h1>
<p>Eval last {args.last_fraction*100:.0f}% sampled, top-k={args.top_k}. Match image: left=GT candidate, right=eval query, green lines=PnP inliers.</p>
<section><h2>Summary</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre></section>
<section><h2>All Samples</h2><table><tr><th>#</th><th>time</th><th>progress</th><th>best rank</th><th>sim</th><th>matches</th><th>landmarks</th><th>inliers</th><th>ratio</th><th>trans err</th><th>dz</th></tr>{table_rows}</table></section>
{''.join(image_cards)}
</body></html>"""
    (output_dir / "index.html").write_text(html_text)


if __name__ == "__main__":
    main()

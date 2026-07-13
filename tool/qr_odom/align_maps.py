#!/usr/bin/env python3
"""
tool/qr_odom/align_maps.py

Solve the rigid transform between two maps' own coordinate frames from the
AprilTag anchor boards they share (see map_anchor_poses.py), via Kabsch/SVD
point registration on the anchors' positions. Pure numpy — no ROS required.

Usage:
  python tool/qr_odom/align_maps.py \
      --a tinynav_db/map_a/anchor_poses.json \
      --b tinynav_db/map_b/anchor_poses.json \
      --out tinynav_db/qrcode/T_map_a_map_b.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the rigid (rotation + translation, no scale) transform minimizing
    sum ||Q_i - (R @ P_i + t)||^2 over corresponding point sets P, Q (N, 3), N >= 3.

    Returns R (3x3), t (3,) such that Q_i ~= R @ P_i + t.
    """
    assert P.shape == Q.shape and P.shape[0] >= 3, "need >=3 corresponding points"
    p_mean = P.mean(axis=0)
    q_mean = Q.mean(axis=0)
    P_c = P - p_mean
    Q_c = Q - q_mean
    H = P_c.T @ Q_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean
    return R, t


def align(anchors_a: dict, anchors_b: dict) -> dict:
    """anchors_a/b: label -> {"T_map_board": 4x4, ...}, as written by map_anchor_poses.py."""
    common = sorted(set(anchors_a) & set(anchors_b))
    if len(common) < 3:
        raise ValueError(
            f"Need >=3 common anchors for a rigid solve, found {len(common)}: {common}"
        )

    P_b = np.array([anchors_b[label]["T_map_board"] for label in common])[:, :3, 3]
    P_a = np.array([anchors_a[label]["T_map_board"] for label in common])[:, :3, 3]

    R, t = kabsch(P_b, P_a)   # P_a ~= R @ P_b + t
    T_a_b = np.eye(4)
    T_a_b[:3, :3] = R
    T_a_b[:3, 3] = t

    residuals_m = {
        label: float(np.linalg.norm(p_a - (R @ p_b + t)))
        for label, p_a, p_b in zip(common, P_a, P_b)
    }

    return {
        "T_map_a_map_b": T_a_b.tolist(),
        "T_map_b_map_a": np.linalg.inv(T_a_b).tolist(),
        "anchors_used": common,
        "residuals_m": residuals_m,
        "rms_residual_m": float(np.sqrt(np.mean(np.square(list(residuals_m.values()))))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Solve the rigid transform between two maps from shared AprilTag anchors."
    )
    parser.add_argument("--a", type=Path, required=True, help="anchor_poses.json for map_a")
    parser.add_argument("--b", type=Path, required=True, help="anchor_poses.json for map_b")
    parser.add_argument("--out", type=Path, default=Path("tinynav_db/qrcode/T_map_a_map_b.json"))
    args = parser.parse_args()

    anchors_a = json.loads(args.a.read_text())
    anchors_b = json.loads(args.b.read_text())

    result = align(anchors_a, anchors_b)

    print(f"Anchors used ({len(result['anchors_used'])}): {result['anchors_used']}")
    for label, res_m in result["residuals_m"].items():
        print(f"  {label}: residual = {res_m * 1000:.1f} mm")
    print(f"RMS residual: {result['rms_residual_m'] * 1000:.1f} mm")
    if result["rms_residual_m"] > 0.05:
        print("WARNING: RMS residual > 50mm — check anchor detections/board placement before trusting this transform.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote T_map_a_map_b -> {args.out}")


if __name__ == "__main__":
    main()

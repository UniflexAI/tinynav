import asyncio
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.foundation_stereo_trt import FoundationStereoTRT


def _load_calib(calib_path: str):
    with open(calib_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 5:
        raise RuntimeError(f"Unexpected calib file format in {calib_path}")

    k_rows = []
    for i in range(1, 4):
        k_rows.append([float(v) for v in lines[i].split()])
    k = np.array(k_rows, dtype=np.float32)

    baseline_line = lines[-1]
    _, val_str = baseline_line.split(":", 1)
    baseline = float(val_str.strip())
    return k, baseline


def _save_visuals(data_dir: str, disp: np.ndarray, depth: np.ndarray):
    np.save(os.path.join(data_dir, "foundation_disp.npy"), disp)
    np.save(os.path.join(data_dir, "foundation_depth.npy"), depth)

    if np.isfinite(disp).any():
        dmin = np.nanmin(disp[np.isfinite(disp)])
        dmax = np.nanmax(disp[np.isfinite(disp)])
        if dmax > dmin:
            disp_norm = (disp - dmin) / (dmax - dmin)
        else:
            disp_norm = np.zeros_like(disp, dtype=np.float32)
    else:
        disp_norm = np.zeros_like(disp, dtype=np.float32)
    disp_u8 = np.clip(disp_norm * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(data_dir, "foundation_disp_vis.png"),
        cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO),
    )

    valid = np.isfinite(depth) & (depth > 0)
    if valid.any():
        zmin = np.nanmin(depth[valid])
        zmax = np.nanmax(depth[valid])
        if zmax > zmin:
            depth_norm = (np.clip(depth, zmin, zmax) - zmin) / (zmax - zmin)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(
        os.path.join(data_dir, "foundation_depth_vis.png"),
        cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS),
    )


def _run_foundation_case(data_dir: str):
    left_path = os.path.join(data_dir, "left.png")
    right_path = os.path.join(data_dir, "right.png")
    calib_path = os.path.join(data_dir, "calib.txt")

    assert os.path.exists(left_path), f"Missing left image: {left_path}"
    assert os.path.exists(right_path), f"Missing right image: {right_path}"
    assert os.path.exists(calib_path), f"Missing calib file: {calib_path}"

    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    assert left is not None and right is not None, "Failed to read stereo images"
    assert left.shape == right.shape, f"Left/right shapes mismatch: {left.shape} vs {right.shape}"

    k, baseline = _load_calib(calib_path)
    fx = float(k[0, 0])

    stereo_engine = FoundationStereoTRT()
    disp, depth = asyncio.run(
        stereo_engine.infer(
            left,
            right,
            np.array([[baseline]], dtype=np.float32),
            np.array([[fx]], dtype=np.float32),
        )
    )

    assert disp.shape == left.shape, f"Disparity shape mismatch: {disp.shape} vs {left.shape}"
    assert depth.shape == left.shape, f"Depth shape mismatch: {depth.shape} vs {left.shape}"
    assert np.isfinite(disp).any(), "No finite disparity values"
    assert np.isfinite(depth).any(), "No finite depth values"

    _save_visuals(data_dir, disp, depth)


def test_foundation_stereo_trt_with_looper_data():
    _run_foundation_case("/tinynav/tests/data/looper")


def test_foundation_stereo_trt_with_realsense_data():
    _run_foundation_case("/tinynav/tests/data/realsense")


if __name__ == "__main__":
    test_foundation_stereo_trt_with_looper_data()
    print("FoundationStereo TRT looper test passed.")
    test_foundation_stereo_trt_with_realsense_data()
    print("FoundationStereo TRT realsense test passed.")

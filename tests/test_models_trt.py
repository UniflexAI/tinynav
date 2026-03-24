import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
import numpy as np
import platform
from codetiming import Timer
from tinynav.core.models_trt import SuperPointTRT, LightGlueTRT, StereoEngineTRT
import asyncio
import cv2

def test_superpoint_trt_with_cache():
    superpoint = SuperPointTRT()
    # Create dummy zero inputs

    # read from /tinynav/tests/data/000000_gray.png
    dummy_image = cv2.imread("/tinynav/tests/data/000000_crop_gray.png", cv2.IMREAD_GRAYSCALE)

    for _ in range(5):
        extract_result_origin = asyncio.run(superpoint.infer(dummy_image))
        with Timer(text="[superpoint infer] Elapsed time: {milliseconds:.02f} ms"):
            extract_result_first = asyncio.run(superpoint.infer(dummy_image))

    assert np.array_equal(extract_result_origin['kpts'], extract_result_first['kpts']), "Cached first kpts result does not match original result."
    assert np.array_equal(extract_result_origin['descps'], extract_result_first['descps']), "Cached first descps result does not match original result."

def test_lightglue_trt_with_cache():
    lightglue = LightGlueTRT()

    dummy_image_0 = cv2.imread("/tinynav/tests/data/000000_crop_gray.png", cv2.IMREAD_GRAYSCALE)
    dummy_image_1 = cv2.imread("/tinynav/tests/data/000001_crop_gray.png", cv2.IMREAD_GRAYSCALE)

    superpoint = SuperPointTRT()
    extract_result_0 = asyncio.run(superpoint.infer(dummy_image_0))
    extract_result_1 = asyncio.run(superpoint.infer(dummy_image_1))
    kpts0 = extract_result_0['kpts']
    descps0 = extract_result_0['descps']
    mask0 = extract_result_0['mask']
    kpts1 = extract_result_1['kpts']
    descps1 = extract_result_1['descps']
    mask1 = extract_result_1['mask']
    for _ in range(5):
        match_result_origin = asyncio.run(lightglue.infer(kpts0, kpts1, descps0, descps1, mask0, mask1, dummy_image_0.shape, dummy_image_1.shape))
        with Timer(text="[lightglue memorized_infer] Elapsed time: {milliseconds:.02f} ms"):
            match_result = asyncio.run(lightglue.infer(kpts0, kpts1, descps0, descps1, mask0, mask1, dummy_image_0.shape, dummy_image_1.shape))

    # vis
    prev_keypoints = kpts0[0]  # (n, 2)
    current_keypoints = kpts1[0]  # (n, 2)
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    kpt_pre = prev_keypoints[valid_mask]
    kpt_cur = current_keypoints[match_indices[valid_mask]]
    # draw matches
    matched_image = cv2.drawMatches(
        dummy_image_0, 
        [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpt_pre],
        dummy_image_1,
        [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpt_cur],
        [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) for i in range(0, len(match_indices[valid_mask]), 4)],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=1
    )
    cv2.imwrite("/tinynav/tests/data/000000-000001_matches.png", matched_image)
    print(f"Number of matches: {len(kpt_pre)}")


def _superpoint_lightglue_matches(
    img0: np.ndarray, img1: np.ndarray, output_path: str
) -> None:
    """
    Run SuperPoint + LightGlue on two grayscale images and write a match
    visualization PNG to output_path.
    """
    superpoint = SuperPointTRT()
    lightglue = LightGlueTRT()

    extract_result_0 = asyncio.run(superpoint.infer(img0))
    extract_result_1 = asyncio.run(superpoint.infer(img1))
    kpts0 = extract_result_0["kpts"]
    descps0 = extract_result_0["descps"]
    mask0 = extract_result_0["mask"]
    kpts1 = extract_result_1["kpts"]
    descps1 = extract_result_1["descps"]
    mask1 = extract_result_1["mask"]

    match_result = asyncio.run(
        lightglue.infer(
            kpts0,
            kpts1,
            descps0,
            descps1,
            mask0,
            mask1,
            img0.shape,
            img1.shape,
        )
    )

    prev_keypoints = kpts0[0]  # (n, 2)
    current_keypoints = kpts1[0]  # (n, 2)
    match_indices = match_result["match_indices"][0]
    valid_mask = match_indices != -1
    kpt_pre = prev_keypoints[valid_mask]
    kpt_cur = current_keypoints[match_indices[valid_mask]]

    matched_image = cv2.drawMatches(
        img0,
        [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpt_pre],
        img1,
        [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpt_cur],
        [
            cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0)
            for i in range(0, len(kpt_pre), 4)
        ],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=1,
    )
    cv2.imwrite(output_path, matched_image)
    print(f"Saved matches visualization to {output_path} (matches: {len(kpt_pre)})")


def test_superpoint_lightglue_looper():
    """
    Run SuperPoint + LightGlue matching on the Looper stereo pair.
    """
    looper_dir = "/tinynav/tests/data/looper"
    left_path = os.path.join(looper_dir, "left.png")
    right_path = os.path.join(looper_dir, "right.png")

    assert os.path.exists(left_path), f"Missing Looper left at {left_path}"
    assert os.path.exists(right_path), f"Missing Looper right at {right_path}"

    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    assert left is not None and right is not None, "Failed to load Looper stereo images"

    assert left.shape == right.shape, "Looper left/right shapes do not match"

    out_path = os.path.join(looper_dir, "matches.png")
    _superpoint_lightglue_matches(left, right, out_path)


def test_superpoint_lightglue_realsense():
    """
    Run SuperPoint + LightGlue matching on the RealSense stereo pair.
    """
    rs_dir = "/tinynav/tests/data/realsense"
    left_path = os.path.join(rs_dir, "left.png")
    right_path = os.path.join(rs_dir, "right.png")

    assert os.path.exists(left_path), f"Missing RealSense left at {left_path}"
    assert os.path.exists(right_path), f"Missing RealSense right at {right_path}"

    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    assert left is not None and right is not None, "Failed to load RealSense stereo images"

    assert left.shape == right.shape, "RealSense left/right shapes do not match"

    out_path = os.path.join(rs_dir, "matches.png")
    _superpoint_lightglue_matches(left, right, out_path)

def _load_looper_calib(calib_path: str):
    """
    Parse calib.txt written by perception_node / extract_stereo_from_rosbag.
    Expected format:
      line 1: header
      line 2-4: 3x3 K matrix (space-separated floats)
      last line: 'baseline (meters): <value>'
    """
    with open(calib_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 5:
        raise RuntimeError(f"Unexpected calib file format in {calib_path}")

    K_rows = []
    for i in range(1, 4):
        parts = lines[i].split()
        K_rows.append([float(v) for v in parts])
    K = np.array(K_rows, dtype=np.float32)

    # Baseline line is expected to be the last non-empty line.
    baseline_line = lines[-1]
    _, val_str = baseline_line.split(":", 1)
    baseline = float(val_str.strip())
    return K, baseline


def test_stereo_engine_trt_with_looper_data():
    """
    Run StereoEngineTRT on the Looper stereo pair stored under tests/data/looper.
    Verifies that disparity/depth are produced with the expected shape and finite values.
    """
    looper_dir = "/tinynav/tests/data/looper"
    left_path = os.path.join(looper_dir, "left.png")
    right_path = os.path.join(looper_dir, "right.png")
    calib_path = os.path.join(looper_dir, "calib.txt")

    assert os.path.exists(left_path), f"Missing left image at {left_path}"
    assert os.path.exists(right_path), f"Missing right image at {right_path}"
    assert os.path.exists(calib_path), f"Missing calib file at {calib_path}"

    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    assert left is not None and right is not None, "Failed to load Looper left/right images"
    assert left.shape == right.shape, "Left/right shapes do not match for Looper data"

    K, baseline = _load_looper_calib(calib_path)
    fx = K[0, 0]

    stereo_engine = StereoEngineTRT()
    disp, depth = asyncio.run(
        stereo_engine.infer(
            left,
            right,
            np.array([[baseline]], dtype=np.float32),
            np.array([[fx]], dtype=np.float32),
        )
    )

    # Shape checks.
    assert disp.shape == left.shape, f"Looper disparity shape {disp.shape} != image shape {left.shape}"
    assert depth.shape == left.shape, f"Looper depth shape {depth.shape} != image shape {left.shape}"

    # Finite checks.
    assert np.isfinite(disp).any(), "Looper disparity has no finite values"
    assert np.isfinite(depth).any(), "Looper depth has no finite values"

    # Save visualizations for Looper outputs.
    disp_vis = disp.copy()
    if np.isfinite(disp_vis).any():
        disp_min = np.nanmin(disp_vis[np.isfinite(disp_vis)])
        disp_max = np.nanmax(disp_vis[np.isfinite(disp_vis)])
        if disp_max > disp_min:
            disp_norm = (disp_vis - disp_min) / (disp_max - disp_min)
        else:
            disp_norm = np.zeros_like(disp_vis, dtype=np.float32)
    else:
        disp_norm = np.zeros_like(disp_vis, dtype=np.float32)
    disp_u8 = np.clip(disp_norm * 255.0, 0, 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(looper_dir, "disp_vis.png"), disp_color)

    depth_vis = depth.copy()
    valid = np.isfinite(depth_vis) & (depth_vis > 0)
    if valid.any():
        depth_min = np.nanmin(depth_vis[valid])
        depth_max = np.nanmax(depth_vis[valid])
        depth_clip = np.clip(depth_vis, depth_min, depth_max)
        depth_norm = (depth_clip - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_vis, dtype=np.float32)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(looper_dir, "depth_vis.png"), depth_color)


def test_stereo_engine_trt_with_realsense_data():
    """
    Run StereoEngineTRT on a RealSense stereo pair stored under tests/data/realsense.
    Verifies that disparity/depth are produced with the expected shape and finite values.
    """
    rs_dir = "/tinynav/tests/data/realsense"
    left_path = os.path.join(rs_dir, "left.png")
    right_path = os.path.join(rs_dir, "right.png")
    calib_path = os.path.join(rs_dir, "calib.txt")

    assert os.path.exists(left_path), f"Missing left image at {left_path}"
    assert os.path.exists(right_path), f"Missing right image at {right_path}"
    assert os.path.exists(calib_path), f"Missing calib file at {calib_path}"

    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    assert left is not None and right is not None, "Failed to load RealSense left/right images"
    assert left.shape == right.shape, "Left/right shapes do not match for RealSense data"

    K, baseline = _load_looper_calib(calib_path)
    fx = K[0, 0]

    stereo_engine = StereoEngineTRT()
    disp, depth = asyncio.run(
        stereo_engine.infer(
            left,
            right,
            np.array([[baseline]], dtype=np.float32),
            np.array([[fx]], dtype=np.float32),
        )
    )

    # Shape checks.
    assert disp.shape == left.shape, f"RealSense disparity shape {disp.shape} != image shape {left.shape}"
    assert depth.shape == left.shape, f"RealSense depth shape {depth.shape} != image shape {left.shape}"

    # Finite checks.
    assert np.isfinite(disp).any(), "RealSense disparity has no finite values"
    assert np.isfinite(depth).any(), "RealSense depth has no finite values"

    # Save visualizations for RealSense outputs.
    disp_vis = disp.copy()
    if np.isfinite(disp_vis).any():
        disp_min = np.nanmin(disp_vis[np.isfinite(disp_vis)])
        disp_max = np.nanmax(disp_vis[np.isfinite(disp_vis)])
        if disp_max > disp_min:
            disp_norm = (disp_vis - disp_min) / (disp_max - disp_min)
        else:
            disp_norm = np.zeros_like(disp_vis, dtype=np.float32)
    else:
        disp_norm = np.zeros_like(disp_vis, dtype=np.float32)
    disp_u8 = np.clip(disp_norm * 255.0, 0, 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_PLASMA)
    cv2.imwrite(os.path.join(rs_dir, "disp_vis.png"), disp_color)

    depth_vis = depth.copy()
    valid = np.isfinite(depth_vis) & (depth_vis > 0)
    if valid.any():
        depth_min = np.nanmin(depth_vis[valid])
        depth_max = np.nanmax(depth_vis[valid])
        depth_clip = np.clip(depth_vis, depth_min, depth_max)
        depth_norm = (depth_clip - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_vis, dtype=np.float32)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(rs_dir, "depth_vis.png"), depth_color)

if __name__ == "__main__":
    test_superpoint_trt_with_cache()
    print("SuperPoint TRT with cache test passed.")
    test_lightglue_trt_with_cache()
    print("LightGlue TRT with cache test passed.")
    test_superpoint_lightglue_looper()
    print("SuperPoint+LightGlue Looper test passed.")
    test_superpoint_lightglue_realsense()
    print("SuperPoint+LightGlue RealSense test passed.")
    test_stereo_engine_trt_with_looper_data()
    print("StereoEngine TRT with Looper data test passed.")
    test_stereo_engine_trt_with_realsense_data()
    print("StereoEngine TRT with RealSense data test passed.")


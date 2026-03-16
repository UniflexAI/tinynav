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
    Verifies that disparity/depth are produced with the expected shape.
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

    # Dynamic-shape engine (default path in StereoEngineTRT).
    stereo_engine_dynamic = StereoEngineTRT()
    disp_dynamic, depth_dynamic = asyncio.run(
        stereo_engine_dynamic.infer(
            left,
            right,
            np.array([[baseline]], dtype=np.float32),
            np.array([[fx]], dtype=np.float32),
        )
    )

    # Constant-shape engine for Looper resolution.
    arch = platform.machine()
    looper_engine_path = f"/tinynav/tinynav/models/retinify_0_1_5_looper_{arch}.plan"
    assert os.path.exists(looper_engine_path), f"Missing constant-shape Looper engine at {looper_engine_path}. Run 'make -C tinynav/models retinify_looper'."

    stereo_engine_const = StereoEngineTRT(engine_path=looper_engine_path)
    disp_const, depth_const = asyncio.run(
        stereo_engine_const.infer(
            left,
            right,
            np.array([[baseline]], dtype=np.float32),
            np.array([[fx]], dtype=np.float32),
        )
    )

    # Save visualizations for dynamic engine outputs.
    disp_vis_dyn = disp_dynamic.copy()
    if np.isfinite(disp_vis_dyn).any():
        disp_min = np.nanmin(disp_vis_dyn[np.isfinite(disp_vis_dyn)])
        disp_max = np.nanmax(disp_vis_dyn[np.isfinite(disp_vis_dyn)])
        if disp_max > disp_min:
            disp_norm_dyn = (disp_vis_dyn - disp_min) / (disp_max - disp_min)
        else:
            disp_norm_dyn = np.zeros_like(disp_vis_dyn, dtype=np.float32)
    else:
        disp_norm_dyn = np.zeros_like(disp_vis_dyn, dtype=np.float32)
    disp_u8_dyn = np.clip(disp_norm_dyn * 255.0, 0, 255).astype(np.uint8)
    disp_color_dyn = cv2.applyColorMap(disp_u8_dyn, cv2.COLORMAP_PLASMA)
    disp_path_dyn = os.path.join(looper_dir, "disp_vis_dynamic.png")
    cv2.imwrite(disp_path_dyn, disp_color_dyn)

    depth_vis_dyn = depth_dynamic.copy()
    valid_dyn = np.isfinite(depth_vis_dyn) & (depth_vis_dyn > 0)
    if valid_dyn.any():
        depth_min = np.nanmin(depth_vis_dyn[valid_dyn])
        depth_max = np.nanmax(depth_vis_dyn[valid_dyn])
        depth_clip_dyn = np.clip(depth_vis_dyn, depth_min, depth_max)
        depth_norm_dyn = (depth_clip_dyn - depth_min) / (depth_max - depth_min)
    else:
        depth_norm_dyn = np.zeros_like(depth_vis_dyn, dtype=np.float32)
    depth_u8_dyn = np.clip(depth_norm_dyn * 255.0, 0, 255).astype(np.uint8)
    depth_color_dyn = cv2.applyColorMap(depth_u8_dyn, cv2.COLORMAP_VIRIDIS)
    depth_path_dyn = os.path.join(looper_dir, "depth_vis_dynamic.png")
    cv2.imwrite(depth_path_dyn, depth_color_dyn)

    # Save visualizations for constant-shape engine outputs.
    disp_vis_const = disp_const.copy()
    if np.isfinite(disp_vis_const).any():
        disp_min_c = np.nanmin(disp_vis_const[np.isfinite(disp_vis_const)])
        disp_max_c = np.nanmax(disp_vis_const[np.isfinite(disp_vis_const)])
        if disp_max_c > disp_min_c:
            disp_norm_const = (disp_vis_const - disp_min_c) / (disp_max_c - disp_min_c)
        else:
            disp_norm_const = np.zeros_like(disp_vis_const, dtype=np.float32)
    else:
        disp_norm_const = np.zeros_like(disp_vis_const, dtype=np.float32)
    disp_u8_const = np.clip(disp_norm_const * 255.0, 0, 255).astype(np.uint8)
    disp_color_const = cv2.applyColorMap(disp_u8_const, cv2.COLORMAP_PLASMA)
    disp_path_const = os.path.join(looper_dir, "disp_vis_const.png")
    cv2.imwrite(disp_path_const, disp_color_const)

    depth_vis_const = depth_const.copy()
    valid_const = np.isfinite(depth_vis_const) & (depth_vis_const > 0)
    if valid_const.any():
        depth_min_c = np.nanmin(depth_vis_const[valid_const])
        depth_max_c = np.nanmax(depth_vis_const[valid_const])
        depth_clip_const = np.clip(depth_vis_const, depth_min_c, depth_max_c)
        depth_norm_const = (depth_clip_const - depth_min_c) / (depth_max_c - depth_min_c)
    else:
        depth_norm_const = np.zeros_like(depth_vis_const, dtype=np.float32)
    depth_u8_const = np.clip(depth_norm_const * 255.0, 0, 255).astype(np.uint8)
    depth_color_const = cv2.applyColorMap(depth_u8_const, cv2.COLORMAP_VIRIDIS)
    depth_path_const = os.path.join(looper_dir, "depth_vis_const.png")
    cv2.imwrite(depth_path_const, depth_color_const)

    # Shape checks for both engines.
    assert disp_dynamic.shape == left.shape, f"Dynamic disparity shape {disp_dynamic.shape} != image shape {left.shape}"
    assert depth_dynamic.shape == left.shape, f"Dynamic depth shape {depth_dynamic.shape} != image shape {left.shape}"
    assert disp_const.shape == left.shape, f"Const disparity shape {disp_const.shape} != image shape {left.shape}"
    assert depth_const.shape == left.shape, f"Const depth shape {depth_const.shape} != image shape {left.shape}"

    # Basic sanity: some finite values should exist.
    assert np.isfinite(disp_dynamic).any(), "Dynamic disparity has no finite values"
    assert np.isfinite(depth_dynamic).any(), "Dynamic depth has no finite values"
    assert np.isfinite(disp_const).any(), "Const disparity has no finite values"
    assert np.isfinite(depth_const).any(), "Const depth has no finite values"

    # Report numeric differences between engines instead of asserting equality,
    # so we always get PNGs even if they differ.
    diff_disp = np.abs(disp_dynamic - disp_const)
    diff_depth = np.abs(depth_dynamic - depth_const)
    print(
        f"[StereoEngine Looper] disp diff mean={diff_disp.mean():.6f}, "
        f"max={diff_disp.max():.6f}"
    )
    print(
        f"[StereoEngine Looper] depth diff mean={diff_depth.mean():.6f}, "
        f"max={diff_depth.max():.6f}"
    )

if __name__ == "__main__":
    test_superpoint_trt_with_cache()
    print("SuperPoint TRT with cache test passed.")
    test_lightglue_trt_with_cache()
    print("LightGlue TRT with cache test passed.")
    test_stereo_engine_trt_with_looper_data()
    print("StereoEngine TRT with Looper data test passed.")


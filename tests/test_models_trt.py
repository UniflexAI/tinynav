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
import yaml

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


def test_superpoint_trt_cache_hit_with_32_salted_images():
    """
    Build 32 salted variants from tests/data/000000.png and run SuperPoint twice.
    Verify second pass hits async LRU cache.
    """
    superpoint = SuperPointTRT()
    superpoint.infer.cache_clear()

    base_path = "/tinynav/tests/data/000000.png"
    base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
    assert base is not None, f"Failed to load {base_path}"

    rng = np.random.default_rng(42)
    images = []
    for _ in range(32):
        noise = rng.integers(-8, 9, size=base.shape, dtype=np.int16)
        salted = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(salted)

    async def _run_two_pass():
        t0 = time.perf_counter()
        first = [await superpoint.infer(img) for img in images]
        first_elapsed_local = time.perf_counter() - t0
        info_first = superpoint.infer.cache_info()

        t1 = time.perf_counter()
        second = [await superpoint.infer(img) for img in images]
        second_elapsed_local = time.perf_counter() - t1
        info_second = superpoint.infer.cache_info()
        return first, second, first_elapsed_local, second_elapsed_local, info_first, info_second

    first_results, second_results, first_elapsed, second_elapsed, info_after_first, info_after_second = asyncio.run(_run_two_pass())

    for r0, r1 in zip(first_results, second_results):
        assert np.array_equal(r0["kpts"], r1["kpts"])
        assert np.array_equal(r0["descps"], r1["descps"])
        assert np.array_equal(r0["mask"], r1["mask"])

    hits_gained = info_after_second.hits - info_after_first.hits
    assert hits_gained >= len(images), (
        f"Expected at least {len(images)} cache hits on second pass, got {hits_gained}. "
        f"cache_info after first={info_after_first}, after second={info_after_second}"
    )

    print(
        f"SuperPoint cache test: first={first_elapsed:.3f}s second={second_elapsed:.3f}s "
        f"hits_gained={hits_gained}"
    )


def test_lightglue_cache_hit_with_consecutive_salted_pairs():
    """
    Follow perception_node infer pattern:
      1) current-frame prev/current matching
      2) sliding-window (_N=5) adjacent-pair association loop
    Run two rounds and verify LightGlue cache hits on round 2.
    """
    superpoint = SuperPointTRT()
    lightglue = LightGlueTRT()
    superpoint.infer.cache_clear()
    lightglue.infer.cache_clear()

    base_path = "/tinynav/tests/data/000000.png"
    base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
    assert base is not None, f"Failed to load {base_path}"

    rng = np.random.default_rng(123)
    images = []
    for _ in range(500):
        noise = rng.integers(-8, 9, size=base.shape, dtype=np.int16)
        salted = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(salted)

    n_window = 5
    keyframe_queue = []
    sp_s = 0.0
    lg_s = 0.0
    match_signatures = []
    sliding_reuse_checks = 0
    sliding_reuse_hits = 0
    seen_pairs = set()
    loop = asyncio.new_event_loop()

    async def _infer_two_images_and_match(frame_idx, new_img):
        nonlocal sliding_reuse_checks, sliding_reuse_hits
        local_signatures = []

        # 1) infer/match previous keyframe with new image
        prev_idx, prev_img = keyframe_queue[-1]
        process_key = (prev_idx, frame_idx)
        r0 = await superpoint.infer(prev_img)
        r1 = await superpoint.infer(new_img)
        m = await lightglue.infer(
            r0["kpts"], r1["kpts"],
            r0["descps"], r1["descps"],
            r0["mask"], r1["mask"],
            prev_img.shape, new_img.shape,
        )
        local_signatures.append(int(np.sum(m["match_indices"] != -1)))
        seen_pairs.add(process_key)

        # 2) append new keyframe into sliding window
        keyframe_queue.append((frame_idx, new_img))
        if len(keyframe_queue) > n_window:
            keyframe_queue.pop(0)

        # 3) sliding-window consecutive pairs
        start = max(0, len(keyframe_queue) - n_window)
        window = keyframe_queue[start:]
        for i in range(0, len(window) - 1):
            info_before_window = lightglue.infer.cache_info()
            idx_a, img_a_w = window[i]
            idx_b, img_b_w = window[i + 1]
            pair_key = (idx_a, idx_b)
            r0 = await superpoint.infer(img_a_w)
            r1 = await superpoint.infer(img_b_w)
            m = await lightglue.infer(
                r0["kpts"], r1["kpts"],
                r0["descps"], r1["descps"],
                r0["mask"], r1["mask"],
                img_a_w.shape, img_b_w.shape,
            )
            local_signatures.append(int(np.sum(m["match_indices"] != -1)))

            if pair_key in seen_pairs:
                sliding_reuse_checks += 1
                window_delta_hit = lightglue.infer.cache_info().hits - info_before_window.hits
                if window_delta_hit >= 1:
                    sliding_reuse_hits += 1
            seen_pairs.add(pair_key)

        return local_signatures

    for frame_idx, img in enumerate(images):
        if len(keyframe_queue) < 1:
            keyframe_queue.append((frame_idx, img))
            continue

        frame_signatures = loop.run_until_complete(_infer_two_images_and_match(frame_idx, img))
        #frame_signatures = asyncio.run(_infer_two_images_and_match(frame_idx, img))
        match_signatures.extend(frame_signatures)

    sig = match_signatures
    info_final = lightglue.infer.cache_info()
    loop.close()

    assert sliding_reuse_checks > 0, "No sliding-window reuse checks were performed."
    assert sliding_reuse_hits == sliding_reuse_checks, (
        f"Expected all reused sliding-window pairs to hit cache, got "
        f"{sliding_reuse_hits}/{sliding_reuse_checks}. cache_info={info_final}"
    )

    print(
        "LightGlue cache test: "
        f"sliding_reuse_hits={sliding_reuse_hits}/{sliding_reuse_checks} "
        f"superpoint_total={sp_s:.3f}s lightglue_total={lg_s:.3f}s "
        f"cache_info={info_final}"
    )

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


def _rectify_euroc_stereo_pair(
    img0: np.ndarray, img1: np.ndarray, cam0_yaml: str, cam1_yaml: str
):
    with open(cam0_yaml, "r") as f:
        cam0 = yaml.safe_load(f)
    with open(cam1_yaml, "r") as f:
        cam1 = yaml.safe_load(f)

    fx0, fy0, cx0, cy0 = cam0["intrinsics"]
    fx1, fy1, cx1, cy1 = cam1["intrinsics"]
    K0 = np.array([[fx0, 0.0, cx0], [0.0, fy0, cy0], [0.0, 0.0, 1.0]], dtype=np.float64)
    K1 = np.array([[fx1, 0.0, cx1], [0.0, fy1, cy1], [0.0, 0.0, 1.0]], dtype=np.float64)
    D0 = np.array(cam0["distortion_coefficients"], dtype=np.float64)
    D1 = np.array(cam1["distortion_coefficients"], dtype=np.float64)

    T_BS0 = np.array(cam0["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    T_BS1 = np.array(cam1["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    T_C0_C1 = np.linalg.inv(T_BS0) @ T_BS1
    R = T_C0_C1[:3, :3]
    T = T_C0_C1[:3, 3]

    h, w = img0.shape
    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
        K0,
        D0,
        K1,
        D1,
        (w, h),
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map0_x, map0_y = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv2.CV_32FC1)
    map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    rect0 = cv2.remap(img0, map0_x, map0_y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)

    fx_rect = float(P0[0, 0])
    baseline_rect = abs(float(P1[0, 3]) / float(P1[0, 0]))
    return rect0, rect1, fx_rect, baseline_rect


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

    # Save raw outputs and visualizations for Looper outputs.
    np.save(os.path.join(looper_dir, "disp.npy"), disp)
    np.save(os.path.join(looper_dir, "depth.npy"), depth)

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

    # Save raw outputs and visualizations for RealSense outputs.
    np.save(os.path.join(rs_dir, "disp.npy"), disp)
    np.save(os.path.join(rs_dir, "depth.npy"), depth)

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


def test_superpoint_and_retinify_with_cam0_cam1():
    """
    Run SuperPoint+LightGlue and Retinify on cam_stereo, and save outputs
    the same way as Looper/RealSense tests.
    """
    stereo_dir = "/tinynav/tests/data/cam_stereo"
    cam0_path = os.path.join(stereo_dir, "cam0.png")
    cam1_path = os.path.join(stereo_dir, "cam1.png")
    cam0_yaml = os.path.join(stereo_dir, "cam0_sensor.yaml")
    cam1_yaml = os.path.join(stereo_dir, "cam1_sensor.yaml")

    assert os.path.exists(cam0_path), f"Missing cam0 image at {cam0_path}"
    assert os.path.exists(cam1_path), f"Missing cam1 image at {cam1_path}"
    assert os.path.exists(cam0_yaml), f"Missing cam0 sensor yaml at {cam0_yaml}"
    assert os.path.exists(cam1_yaml), f"Missing cam1 sensor yaml at {cam1_yaml}"

    cam0 = cv2.imread(cam0_path, cv2.IMREAD_GRAYSCALE)
    cam1 = cv2.imread(cam1_path, cv2.IMREAD_GRAYSCALE)
    assert cam0 is not None and cam1 is not None, "Failed to load cam0/cam1 images"
    assert cam0.shape == cam1.shape, "cam0/cam1 shapes do not match"

    cam0_rect, cam1_rect, fx_rect, baseline_rect = _rectify_euroc_stereo_pair(
        cam0, cam1, cam0_yaml, cam1_yaml
    )
    cv2.imwrite(os.path.join(stereo_dir, "cam0_rect.png"), cam0_rect)
    cv2.imwrite(os.path.join(stereo_dir, "cam1_rect.png"), cam1_rect)

    # SuperPoint + LightGlue visualization on rectified images.
    out_matches = os.path.join(stereo_dir, "matches.png")
    _superpoint_lightglue_matches(cam0_rect, cam1_rect, out_matches)

    # Retinify disparity/depth.
    stereo_engine = StereoEngineTRT()
    disp, depth = asyncio.run(
        stereo_engine.infer(
            cam0_rect,
            cam1_rect,
            np.array([[baseline_rect]], dtype=np.float32),
            np.array([[fx_rect]], dtype=np.float32),
        )
    )

    assert disp.shape == cam0_rect.shape, f"Disparity shape {disp.shape} != input shape {cam0_rect.shape}"
    assert depth.shape == cam0_rect.shape, f"Depth shape {depth.shape} != input shape {cam0_rect.shape}"
    assert np.isfinite(disp).any(), "Retinify disparity has no finite values"
    assert np.isfinite(depth).any(), "Retinify depth has no finite values"

    # Save raw outputs and visualizations.
    np.save(os.path.join(stereo_dir, "disp.npy"), disp)
    np.save(os.path.join(stereo_dir, "depth.npy"), depth)

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
    cv2.imwrite(os.path.join(stereo_dir, "disp_vis.png"), disp_color)

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
    cv2.imwrite(os.path.join(stereo_dir, "depth_vis.png"), depth_color)

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
    test_superpoint_and_retinify_with_cam0_cam1()
    print("SuperPoint+Retinify cam0/cam1 test passed.")

import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
import numpy as np
from codetiming import Timer
from tinynav.core.models_trt import SuperPointTRT, LightGlueTRT
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
    

if __name__ == "__main__":
    test_superpoint_trt_with_cache()
    print("SuperPoint TRT with cache test passed.")
    test_lightglue_trt_with_cache()
    print("LightGlue TRT with cache test passed.")


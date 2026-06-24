import numpy as np

from tinynav.core.perception_node import filter_sparse_stereo_matches


def test_filter_sparse_stereo_matches_keeps_valid_rectified_matches():
    left = np.array([[100.0, 50.0], [130.0, 60.0]], dtype=np.float32)
    right = np.array([[90.0, 50.5], [120.0, 59.5]], dtype=np.float32)
    matches = np.array([0, 1], dtype=np.int32)

    filtered, stats = filter_sparse_stereo_matches(left, right, matches, fx=100.0, baseline=0.1)

    assert np.array_equal(filtered, matches)
    assert stats["raw_matches"] == 2
    assert stats["valid_matches"] == 2


def test_filter_sparse_stereo_matches_rejects_negative_disparity():
    left = np.array([[100.0, 50.0]], dtype=np.float32)
    right = np.array([[101.0, 50.0]], dtype=np.float32)
    matches = np.array([0], dtype=np.int32)

    filtered, stats = filter_sparse_stereo_matches(left, right, matches, fx=100.0, baseline=0.1)

    assert filtered[0] == -1
    assert stats["rejected_disparity"] == 1


def test_filter_sparse_stereo_matches_rejects_vertical_mismatch():
    left = np.array([[100.0, 50.0]], dtype=np.float32)
    right = np.array([[90.0, 53.5]], dtype=np.float32)
    matches = np.array([0], dtype=np.int32)

    filtered, stats = filter_sparse_stereo_matches(left, right, matches, fx=100.0, baseline=0.1)

    assert filtered[0] == -1
    assert stats["rejected_vertical"] == 1


def test_filter_sparse_stereo_matches_rejects_out_of_bounds_indices():
    left = np.array([[100.0, 50.0]], dtype=np.float32)
    right = np.array([[90.0, 50.0]], dtype=np.float32)
    matches = np.array([4], dtype=np.int32)

    filtered, stats = filter_sparse_stereo_matches(left, right, matches, fx=100.0, baseline=0.1)

    assert filtered[0] == -1
    assert stats["rejected_out_of_bounds"] == 1

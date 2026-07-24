import numpy as np

from tinynav.core.depth_reprojection import reproject_depth_to_camera


def make_planar_depth(shape, value=2.0):
    depth = np.zeros(shape, dtype=np.float32)
    depth[shape[0] // 4 : 3 * shape[0] // 4, shape[1] // 4 : 3 * shape[1] // 4] = value
    return depth


def test_identity_transform_reproduces_source():
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    depth = make_planar_depth((64, 64))
    out = reproject_depth_to_camera(depth, K, np.eye(4), K, (64, 64), dilate_kernel_size=0)
    nonzero_src = depth > 0.0
    np.testing.assert_allclose(out[nonzero_src], depth[nonzero_src], atol=1e-4)


def test_translated_camera_shifts_depth_patch():
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    depth = make_planar_depth((64, 64), value=2.0)

    # point_in_dst = point_in_src + translation; a constant +X offset shifts every reprojected
    # pixel by -fx*tx/Z columns (since u = fx*x/z + cx).
    tx = -0.2
    T_dst_from_src = np.eye(4)
    T_dst_from_src[0, 3] = tx

    out = reproject_depth_to_camera(depth, K, T_dst_from_src, K, (64, 64), dilate_kernel_size=0)

    expected_shift_px = round(100.0 * tx / 2.0)  # fx * tx / Z
    src_cols = np.nonzero(depth.any(axis=0))[0]
    dst_cols = np.nonzero(out.any(axis=0))[0]
    assert dst_cols.min() - src_cols.min() == expected_shift_px
    assert dst_cols.max() - src_cols.max() == expected_shift_px


def test_dilation_fills_small_gaps_without_changing_values():
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((64, 64), dtype=np.float32)
    depth[32, 32] = 3.0  # a single valid pixel, simulating sparse forward-splat coverage

    out_no_dilate = reproject_depth_to_camera(depth, K, np.eye(4), K, (64, 64), dilate_kernel_size=0)
    out_dilated = reproject_depth_to_camera(depth, K, np.eye(4), K, (64, 64), dilate_kernel_size=5)

    assert np.count_nonzero(out_no_dilate) == 1
    assert np.count_nonzero(out_dilated) > 1
    assert out_dilated[32, 32] == 3.0


def test_empty_depth_returns_zeros():
    K = np.eye(3)
    depth = np.zeros((16, 16), dtype=np.float32)
    out = reproject_depth_to_camera(depth, K, np.eye(4), K, (16, 16))
    assert np.count_nonzero(out) == 0
    assert out.shape == (16, 16)


if __name__ == "__main__":
    test_identity_transform_reproduces_source()
    test_translated_camera_shifts_depth_patch()
    test_dilation_fills_small_gaps_without_changing_values()
    test_empty_depth_returns_zeros()
    print("All depth reprojection tests passed.")

"""Which camera's intrinsics PnP is given.

PnP projects the map's 3D points into the image the 2D points came from, which is the
LIVE one. `map_K` describes the camera that built the map, and is right only where the
map's own depth is un-projected.

The two were equal until the camera on this rig was replaced. Both calibrations are
correct and describe different hardware, which is also why a map's `intrinsics.npy` must
never be overwritten with today's: it is the record of how that map's depth was
un-projected. Measured on 122 on 2026-09-03: live fx 301.61 / cx 268.37 against every
map's 312.30 / 275.90. **The wrong intrinsics do not cost inliers**, which was the first guess and is wrong:
PnP solves for a pose, and a focal error is absorbed almost exactly by a compensating
one -- a 3.4% focal scaling is nearly a 3.4% depth scaling. RANSAC then fits every
point. What it costs is the POSE: the fix lands at the wrong range and bearing, with
full confidence. So that is what these assert.
"""
import unittest

import cv2
import numpy as np

from tinynav.core.map_node import MapNode

LIVE_K = np.array([[301.6144, 0.0, 268.3674],
                   [0.0, 301.6144, 315.1000],
                   [0.0, 0.0, 1.0]])
MAP_K = np.array([[312.3000, 0.0, 275.9000],
                  [0.0, 312.3000, 315.1000],
                  [0.0, 0.0, 1.0]])


def _correspondences(K, n=200, seed=0):
    """Points spread across the frame, projected with `K` from a known pose. Exact, so
    every inlier lost is the intrinsics and nothing else."""
    rng = np.random.default_rng(seed)
    pts3d = np.column_stack([
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(-1.5, 1.5, n),
        rng.uniform(2.0, 8.0, n),
    ]).astype(np.float64)
    rvec = np.zeros(3)
    tvec = np.array([0.1, -0.05, 0.2])
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, K, None)
    return pts3d, pts2d.reshape(-1, 2).astype(np.float64), tvec


class _Node:
    """Only what `rank_relocalization_candidates` reads."""

    def __init__(self):
        self.K = LIVE_K
        self.map_K = MAP_K


def _recover(K_for_pnp):
    """The pose PnP recovers from correspondences the LIVE camera produced, solved with
    `K_for_pnp`. Returns (inlier ratio, translation error against the truth)."""
    from tinynav.core.math_utils import rerank_by_pnp_inliers
    pts3d, pts2d, truth = _correspondences(LIVE_K)
    ok, pose, ratio, _, inliers, _ = rerank_by_pnp_inliers([(pts3d, pts2d)], K_for_pnp)
    assert ok
    return ratio, float(np.linalg.norm(pose[:3, 3] - truth))


class IntrinsicsTest(unittest.TestCase):
    def test_the_live_intrinsics_recover_the_pose(self):
        ratio, err = _recover(LIVE_K)
        self.assertGreater(ratio, 0.9)
        self.assertLess(err, 0.01, 'the pose was not recovered from exact data')

    def test_the_maps_intrinsics_bias_the_pose_without_losing_confidence(self):
        """This is the shape of the bug: it does not fail, it lies. Full inliers, wrong
        place -- which is why it went unnoticed while every fused pose carried it."""
        ratio, err = _recover(MAP_K)
        self.assertGreater(ratio, 0.9, 'expected the wrong K to still fit every point')
        self.assertGreater(err, 0.05,
                           f'the map intrinsics biased the pose by only {err:.3f}m; '
                           'if a 3.4% focal error is now harmless this test is stale')

    def test_the_live_intrinsics_are_the_ones_rank_uses(self):
        """Through the real method, so a revert to `map_K` shows up here."""
        pts3d, pts2d, truth = _correspondences(LIVE_K)
        ok, pose, _ = MapNode.rank_relocalization_candidates(
            _Node(), [(pts3d, pts2d)], [0])
        self.assertTrue(ok)
        self.assertLess(float(np.linalg.norm(pose[:3, 3] - truth)), 0.01,
                        'rank_relocalization_candidates is not using the live K')


if __name__ == '__main__':
    unittest.main()

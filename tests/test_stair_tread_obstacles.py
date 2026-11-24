import dataclasses
import numpy as np
import pytest

from tinynav.core.planning_node import build_obstacle_map
from tinynav.core.robot_specs import ObstacleConfig
from tinynav.core.stair_obstacles import measured_tread_mask


def scene(slope=.5):
    K = np.array([[75., 0., 67.], [0., 75., 79.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    T[2, 3] = .3
    v, u = np.indices((160, 136))
    denominator = slope + (v - K[1, 2]) / K[1, 1]
    depth = np.divide(.3, denominator, out=np.zeros_like(denominator), where=denominator > 0.)
    depth[(depth < .2) | (depth > 1.8)] = 0.
    grid = np.zeros((80, 80, 30))
    origin = np.array([-1., -2., -.45])
    return depth, T, K, grid, origin


def column(grid, origin, x, y, low, high):
    ij = np.floor((np.array([x, y]) - origin[:2]) / .05).astype(int)
    z = origin[2] + (np.arange(grid.shape[2]) + .5) * .05
    grid[tuple(ij)] = np.where((z >= low) & (z <= high), .2, 0.)
    return tuple(ij)


def test_measured_rising_support_removes_step_curtain_but_preserves_low_barrier_and_wall():
    depth, T, K, grid, origin = scene()
    tread = column(grid, origin, .925, -.125, .10, .48)
    barrier = column(grid, origin, .925, .125, .10, .72)
    wall = column(grid, origin, 1.125, .325, -.1, 1.)
    support = measured_tread_mask(depth, T, K, grid, origin, .05, .3)
    cfg = ObstacleConfig(dilation_cells=0)
    before = build_obstacle_map(grid, origin, .05, .3, cfg)
    after = build_obstacle_map(grid, origin, .05, .3, cfg, support)
    assert before[tread] and not after[tread]
    assert after[barrier] and after[wall]
    assert not support[barrier] and not support[wall]
    assert np.all(after <= before)
    assert np.array_equal(build_obstacle_map(grid, origin, .05, .3, cfg), before)
    inflated = build_obstacle_map(grid, origin, .05, .3, dataclasses.replace(cfg, dilation_cells=1), support)
    assert inflated[wall] and inflated[wall[0], wall[1] + 1]


@pytest.mark.parametrize('slope', [0., -.5, 1.])
def test_no_upstairs_evidence_does_not_clear_obstacles(slope):
    depth, T, K, grid, origin = scene(slope)
    column(grid, origin, .925, -.125, .10, .48)
    assert not measured_tread_mask(depth, T, K, grid, origin, .05, .3).any()


def test_missing_support_and_clipped_wall_are_retained():
    depth, T, K, grid, origin = scene()
    wall = column(grid, origin, 1.125, .025, -.1, 1.1)
    assert not measured_tread_mask(depth, T, K, grid, origin, .05, .3)[wall]
    assert not measured_tread_mask(np.zeros_like(depth), T, K, grid, origin, .05, .3).any()


def test_tread_filter_is_world_heading_invariant():
    depth, T, K, grid, origin = scene()
    ij = column(grid, origin, .925, -.125, .10, .48)
    reference = measured_tread_mask(depth, T, K, grid, origin, .05, .3)
    # Rotate by 180 degrees around the voxel grid center, including its camera pose.
    center = origin[:2] + np.array(grid.shape[:2]) * .05 / 2
    rotated = T.copy()
    rotated[:2, :3] *= -1.
    rotated[:2, 3] = 2 * center - T[:2, 3]
    actual = measured_tread_mask(depth, rotated, K, grid[::-1, ::-1].copy(), origin, .05, .3)
    assert reference[ij]
    occupied = np.any(grid > .1, axis=2)
    assert np.array_equal(actual[::-1, ::-1] & occupied, reference & occupied)

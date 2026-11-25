import dataclasses
import numpy as np
import pytest

from tinynav.core.planning_node import build_obstacle_map
from tinynav.core.robot_specs import ObstacleConfig
from tinynav.core.stair_obstacles import measured_tread_mask


def scene(slope=.5, bearing=0.):
    K = np.array([[75., 0., 67.], [0., 75., 79.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    T[2, 3] = .3
    v, u = np.indices((160, 136))
    angle = np.deg2rad(bearing)
    denominator = slope * np.cos(angle) - slope * np.sin(angle) * (u - K[0, 2]) / K[0, 0] + (v - K[1, 2]) / K[1, 1]
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


@pytest.mark.parametrize('bearing', [-60., -32., 32., 60.])
@pytest.mark.parametrize('partial', [False, True])
def test_oblique_and_partially_visible_treads_keep_obstacles(bearing, partial):
    depth, T, K, grid, origin = scene(bearing=bearing)
    if partial:
        depth[:35] = 0.
        depth[110:] = 0.
        depth[:, :35] = 0.
    support = measured_tread_mask(depth, T, K, grid, origin, .05, .3)
    assert support.any()
    xy = origin[:2] + (np.indices(grid.shape[:2]).transpose(1, 2, 0) + .5) * .05
    ground = .5 * (xy[..., 0] * np.cos(np.deg2rad(bearing)) + xy[..., 1] * np.sin(np.deg2rad(bearing)))
    choices = np.argwhere(support & (ground > .39) & (ground < .47))
    assert len(choices)
    i, j = choices[len(choices) // 2]
    x, y = xy[i, j]
    cell = column(grid, origin, x, y, .1, ground[i, j] + .02)
    support = measured_tread_mask(depth, T, K, grid, origin, .05, .3)
    assert support[cell]
    column(grid, origin, x, y, .1, ground[i, j] + .3)
    assert not measured_tread_mask(depth, T, K, grid, origin, .05, .3)[cell]


def test_projective_clearing_protects_observed_bars_occlusion_and_unknown():
    from tinynav.core.stair_obstacles import projective_free_mask
    K = np.array([[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]])
    T = np.eye(4)
    origin = np.array([-.25, -.25, -.75])
    grid = np.zeros((10, 10, 60))
    cell = (5, 5, 25)  # z = .525m, inside the image
    grid[cell] = .2
    depth = np.full((64, 64), 2., dtype=np.float32)
    assert projective_free_mask(depth, T, K, grid, origin, .05)[cell]
    depth[35, 35] = .5  # a thin bar anywhere in the projected footprint protects the voxel
    assert not projective_free_mask(depth, T, K, grid, origin, .05)[cell]
    depth[35, 35] = 0.
    assert not projective_free_mask(depth, T, K, grid, origin, .05)[cell]
    depth.fill(.4)  # nearer surface occludes the voxel
    assert not projective_free_mask(depth, T, K, grid, origin, .05)[cell]
    depth.fill(2.)
    grid.fill(0.); grid[5, 5, 3] = .2  # behind the camera
    assert not projective_free_mask(depth, T, K, grid, origin, .05).any()


def test_projective_clearing_needs_repeated_evidence_and_does_not_override_hits(monkeypatch):
    from tinynav.core import planning_node
    from types import SimpleNamespace
    grid = np.zeros((10, 10, 30)); grid[5, 5, 25] = .2
    fresh = np.zeros_like(grid)
    node = SimpleNamespace(origin=np.array([-.25, -.25, -.75]), grid_shape=grid.shape, resolution=.05,
        step=10, occupancy_grid=grid, in_stair_mode=True, _stair_up=True, obstacle_config=ObstacleConfig())
    monkeypatch.setattr(planning_node, 'run_raycasting_loopy', lambda *a: fresh.copy())
    depth = np.full((64, 64), 2., dtype=np.float32)
    planning_node.PlanningNode.update_occupancy_grid(node, depth, np.eye(4), 50., 50., 32., 32.)
    assert node.occupancy_grid[5, 5, 25] > .1
    planning_node.PlanningNode.update_occupancy_grid(node, depth, np.eye(4), 50., 50., 32., 32.)
    assert node.occupancy_grid[5, 5, 25] < .1
    node.occupancy_grid[5, 5, 25] = .2
    fresh[5, 5, 25] = .1
    planning_node.PlanningNode.update_occupancy_grid(node, depth, np.eye(4), 50., 50., 32., 32.)
    assert node.occupancy_grid[5, 5, 25] == .2

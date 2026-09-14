import os
import sys

import numpy as np
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinynav.core.planning_node import (
    camera_to_robot_center,
    grid_from_msg,
    grid_origin,
    mask_from_msg,
    mask_plane_z,
    obstacle_mask_msg,
    occupancy_3d_msg,
)
from tinynav.core.planning_vis_node import footprint_points, voxel_points
from tinynav.core.robot_specs import ROBOT_CONFIG

# Non-square on purpose: a swapped axis or a row/column-major mix-up only shows on X != Y.
X, Y, Z = 40, 25, 7
RESOLUTION = 0.05
ORIGIN = np.array([-1.23, 4.56, -0.35])
STAMP = Time(sec=12, nanosec=345)


def _grid():
    return np.clip(np.random.default_rng(0).normal(0, 0.1, (X, Y, Z)), -0.2, 0.2)


def _sent_mask(mask):
    return obstacle_mask_msg(mask, ORIGIN, RESOLUTION, mask_plane_z(ORIGIN[2], Z, RESOLUTION), STAMP)


def test_mask_arrives_as_planning_holds_it():
    mask = _grid().max(axis=2) > 0.15
    got, resolution, _ = mask_from_msg(_sent_mask(mask))
    assert got.shape == (X, Y)
    assert np.array_equal(got, mask)
    assert resolution == RESOLUTION


def test_grid_arrives_bit_for_bit():
    grid = _grid()
    got = grid_from_msg(occupancy_3d_msg(grid, Header(stamp=STAMP)))
    assert got.shape == (X, Y, Z)
    assert np.array_equal(got, grid)


def test_a_voxel_lands_on_its_cell():
    grid = np.zeros((X, Y, Z))
    cell = (31, 4, 5)
    grid[cell] = 0.2
    _, resolution, mask_origin = mask_from_msg(_sent_mask(np.zeros((X, Y), bool)))
    origin = grid_origin(mask_origin, resolution, Z)
    points = voxel_points(grid_from_msg(occupancy_3d_msg(grid, Header(stamp=STAMP))), origin, resolution)
    assert points.shape == (1, 3)
    assert np.allclose(points[0], ORIGIN + np.array(cell) * RESOLUTION, atol=1e-9)


def test_footprint_outlines_the_control_rectangle():
    yaw = 0.7
    T = np.eye(4)
    # camera convention: body z is forward, x is left
    T[:3, :3] = [[np.cos(yaw), 0, -np.sin(yaw)], [np.sin(yaw), 0, np.cos(yaw)], [0, 1, 0]]
    T[:3, 3] = [2.0, -1.0, 0.3]
    pts = footprint_points(T)
    fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
    rel = pts - camera_to_robot_center(T)
    along, across = rel @ T[:3, 2], rel @ T[:3, 0]
    assert len(pts) == 84
    assert np.isclose(along.max(), fl) and np.isclose(along.min(), -rl)
    assert np.isclose(np.abs(across).max(), hw)
    on_edge = np.isclose(along, fl) | np.isclose(along, -rl) | np.isclose(np.abs(across), hw)
    assert on_edge.all()


if __name__ == "__main__":
    test_mask_arrives_as_planning_holds_it()
    test_grid_arrives_bit_for_bit()
    test_a_voxel_lands_on_its_cell()
    test_footprint_outlines_the_control_rectangle()

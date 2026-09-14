import os
import sys
import types

import numpy as np
from builtin_interfaces.msg import Time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinynav.core.planning_node import PlanningNode
from tinynav.core.planning_vis_node import decode_mask, footprint_points
from tinynav.core.robot_specs import ROBOT_CONFIG


class _Capture:
    def publish(self, msg):
        self.msg = msg


def test_mask_arrives_as_planning_holds_it():
    # non-square, so a swapped axis or a row/column-major mix-up shows
    X, Y, Z, resolution = 40, 25, 7, 0.05
    origin = np.array([-1.23, 4.56, -0.35])
    mask = np.random.default_rng(0).random((X, Y)) > 0.7
    planning = types.SimpleNamespace(origin=origin, grid_shape=(X, Y, Z), resolution=resolution,
                                     obstacle_mask_pub=_Capture())
    PlanningNode.publish_obstacle_mask(planning, mask, Time(sec=1))
    got, got_resolution, plane = decode_mask(planning.obstacle_mask_pub.msg)
    assert np.array_equal(got, mask)
    assert got_resolution == resolution
    assert np.allclose(plane, origin + [0.0, 0.0, Z * resolution / 2])


def test_footprint_outlines_the_control_rectangle():
    yaw = 0.7
    T = np.eye(4)
    # camera convention: body z is forward, x is left
    T[:3, :3] = [[np.cos(yaw), 0, -np.sin(yaw)], [np.sin(yaw), 0, np.cos(yaw)], [0, 1, 0]]
    T[:3, 3] = [2.0, -1.0, 0.3]
    pts = footprint_points(T)
    fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
    rel = pts - (T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d)
    along, across = rel @ T[:3, 2], rel @ T[:3, 0]
    assert len(pts) == 84
    assert np.isclose(along.max(), fl) and np.isclose(along.min(), -rl) and np.isclose(np.abs(across).max(), hw)
    assert (np.isclose(along, fl) | np.isclose(along, -rl) | np.isclose(np.abs(across), hw)).all()


if __name__ == "__main__":
    test_mask_arrives_as_planning_holds_it()
    test_footprint_outlines_the_control_rectangle()

import os
import sys
import types
from dataclasses import replace
from unittest import mock

import numpy as np
from builtin_interfaces.msg import Time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinynav.core.planning_node import PlanningNode
from tinynav.core import planning_vis_node
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
    fl, rl, hw = ROBOT_CONFIG.collision_footprint()
    rel = pts - (T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d)
    along, across = rel @ T[:3, 2], rel @ T[:3, 0]
    assert len(pts) == 84
    assert np.isclose(along.max(), fl) and np.isclose(along.min(), -rl) and np.isclose(np.abs(across).max(), hw)
    assert (np.isclose(along, fl) | np.isclose(along, -rl) | np.isclose(np.abs(across), hw)).all()


def test_the_footprint_drawn_is_the_one_checked():
    """At footprint_scale 0.8 the operator sees the 0.8 outline planning checks, around
    the same centre: the camera offset is not the scale's."""
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    body = ROBOT_CONFIG.footprint_from_control()
    with mock.patch.object(planning_vis_node, 'ROBOT_CONFIG',
                           replace(ROBOT_CONFIG, footprint_scale=(1.0, 1.0, 1.0))):
        full = footprint_points(T)
    with mock.patch.object(planning_vis_node, 'ROBOT_CONFIG',
                           replace(ROBOT_CONFIG, footprint_scale=(0.8, 0.8, 0.8))):
        scaled = footprint_points(T)
    center = T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d
    assert np.allclose(full[:, 0].max() - center[0], body[0])
    assert np.allclose(scaled - center, 0.8 * (full - center))
    with mock.patch.object(planning_vis_node, 'ROBOT_CONFIG',
                           replace(ROBOT_CONFIG, footprint_scale=(1.0, 0.6, 0.7))):
        rel = footprint_points(T) - center
    # this T: body forward is world +x, body left is world +y
    assert np.isclose(rel[:, 0].max(), body[0]) and np.isclose(rel[:, 0].min(), -0.6 * body[1])
    assert np.isclose(np.abs(rel[:, 1]).max(), 0.7 * body[2])


def test_the_outline_follows_the_factors_planning_says_it_uses():
    """A map's factors arrive on planning's in-use topics, one per direction; the
    outline is drawn at them, so the operator never sees the rig's outline while
    planning checks the map's."""
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    rig = replace(ROBOT_CONFIG, footprint_scale=(1.0, 1.0, 1.0))
    with mock.patch.object(planning_vis_node, 'ROBOT_CONFIG', rig):
        full = footprint_points(T)
        at = footprint_points(T, scale=(1.0, 0.7, 0.6))
    center = T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d
    fl, rl, hw = ROBOT_CONFIG.footprint_from_control()
    rel = at - center
    assert np.isclose(rel[:, 0].max(), fl) and np.isclose(rel[:, 0].min(), -0.7 * rl)
    assert np.isclose(np.abs(rel[:, 1]).max(), 0.6 * hw)

    vis = types.SimpleNamespace(scale_in_use=None)
    say = planning_vis_node.PlanningVisNode.scale_in_use_callback
    with mock.patch.object(planning_vis_node, 'ROBOT_CONFIG', rig):
        say(vis, 'rear', types.SimpleNamespace(data=float(np.float32(0.7))))
        assert vis.scale_in_use == (1.0, 0.7, 1.0), 'the directions not said yet are the rig\'s'
        say(vis, 'width', types.SimpleNamespace(data=float(np.float32(0.6))))
        assert vis.scale_in_use == (1.0, 0.7, 0.6)
        for not_a_factor in (float('nan'), 0.2, 1.5):
            say(vis, 'width', types.SimpleNamespace(data=not_a_factor))
            assert vis.scale_in_use == (1.0, 0.7, 0.6), not_a_factor

        vis.footprint_pub = types.SimpleNamespace(get_subscription_count=lambda: 1,
                                                  publish=lambda msg: setattr(vis, 'cloud', msg))
        odom = mock.Mock()
        odom.header.stamp = Time(sec=1)
        with mock.patch.object(planning_vis_node, 'msg2np', return_value=(T, None)):
            planning_vis_node.PlanningVisNode.odom_callback(vis, odom)
    drawn = np.array([[p.x, p.y, p.z] for p in vis.cloud.points])
    assert np.allclose(drawn, at, atol=1e-5)
    assert not np.allclose(drawn, full, atol=1e-3)


if __name__ == "__main__":
    test_mask_arrives_as_planning_holds_it()
    test_footprint_outlines_the_control_rectangle()
    test_the_footprint_drawn_is_the_one_checked()
    test_the_outline_follows_the_factors_planning_says_it_uses()

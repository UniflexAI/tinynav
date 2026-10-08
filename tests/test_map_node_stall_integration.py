"""Exercises MapNode.nav_target_timer_callback's replan/stall decision directly,
without booting SLAM/vision, TF, or loading a real built map.

Unlike test_map_node.py (which avoids importing tinynav.core.map_node because it
pulls in TensorRT/rclpy at module load), this file does import it, so it needs the
full tinynav environment. Run it inside the uniflexai/tinynav container, e.g.:

    docker exec <container> bash -lc '
      source /opt/ros/humble/setup.bash
      cd /tinynav && uv run --python /opt/venv/bin/python pytest \
        tests/test_map_node_stall_integration.py -v
    '

MapNode is built via __new__ (skipping __init__, which loads TensorRT models and a
map directory) and hand-filled with only the state nav_target_timer_callback reads
or writes. generate_nav_path_in_map -- the expensive SDF search over a real map --
is stubbed out with a canned straight-line path, since this test is about the
replan/stall decision, not the path search itself.
"""
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from builtin_interfaces.msg import Time

sys.path.insert(0, ".")
from tinynav.core.map_node import MapNode


def make_bare_map_node(poi_xyz, start_xyz):
    node = MapNode.__new__(MapNode)
    node.get_logger = MagicMock(return_value=MagicMock())
    node.get_clock = MagicMock(return_value=MagicMock(
        now=MagicMock(return_value=MagicMock(to_msg=MagicMock(return_value=Time())))
    ))
    for pub in (
        "current_pose_in_map_pub", "nav_progress_pub", "poi_change_pub",
        "nav_done_pub", "global_plan_pub", "target_pose_pub",
    ):
        setattr(node, pub, MagicMock())
    node.tf_broadcaster = MagicMock()

    node.T_from_map_to_odom = np.eye(4)
    node.latest_odom_pose = np.eye(4)
    node.latest_odom_pose[:3, 3] = start_xyz
    node.pois = {0: np.array(poi_xyz)}
    node.poi_index = 0
    node._nav_completed = False
    node.cached_nav_path_in_map = None
    node.cached_nav_path_poi_index = -1
    node._leg_initial_length = None
    node._leg_start_time = None
    node._speed_estimate = None
    node._last_progress_remaining = None
    node._last_progress_time = None
    node.stall_progress_eps = 0.05
    node.stall_timeout_s = 8.0
    return node


def straight_path(start_xyz, goal_xyz, n=51):
    start = np.asarray(start_xyz, dtype=float)
    goal = np.asarray(goal_xyz, dtype=float)
    return np.array([start + (goal - start) * t for t in np.linspace(0.0, 1.0, n)])


def test_stall_forces_a_replan_when_stuck_on_path():
    poi = [5.0, 0.0, 0.0]
    node = make_bare_map_node(poi_xyz=poi, start_xyz=[0.0, 0.0, 0.0])
    node.generate_nav_path_in_map = MagicMock(return_value=straight_path([0.0, 0.0, 0.0], poi))

    with patch("tinynav.core.map_node.time.time") as fake_time:
        fake_time.return_value = 0.0
        node.nav_target_timer_callback()
        assert node.generate_nav_path_in_map.call_count == 1  # first tick always replans

        fake_time.return_value = 1.0  # robot hasn't moved; well under the timeout
        node.nav_target_timer_callback()
        assert node.generate_nav_path_in_map.call_count == 1  # no replan yet

        fake_time.return_value = 10.0  # still hasn't moved; past stall_timeout_s
        node.nav_target_timer_callback()
        assert node.generate_nav_path_in_map.call_count == 2  # stall forced a replan
        node.get_logger().warning.assert_called()
        assert node._last_progress_remaining is None  # reset after the forced replan


def test_steady_progress_never_stalls():
    poi = [5.0, 0.0, 0.0]
    node = make_bare_map_node(poi_xyz=poi, start_xyz=[0.0, 0.0, 0.0])
    node.generate_nav_path_in_map = MagicMock(return_value=straight_path([0.0, 0.0, 0.0], poi))

    with patch("tinynav.core.map_node.time.time") as fake_time:
        fake_time.return_value = 0.0
        node.nav_target_timer_callback()
        assert node.generate_nav_path_in_map.call_count == 1

        # advance both the clock (past stall_timeout_s each tick) and the robot's
        # position (real progress along the path) every tick -- should never stall
        for i in range(1, 6):
            fake_time.return_value = float(i) * 10.0
            node.latest_odom_pose[:3, 3] = [float(i) * 0.5, 0.0, 0.0]
            node.nav_target_timer_callback()

        assert node.generate_nav_path_in_map.call_count == 1  # never re-triggered
        assert node._last_progress_remaining is not None


if __name__ == "__main__":
    test_stall_forces_a_replan_when_stuck_on_path()
    print("Stall integration test passed.")
    test_steady_progress_never_stalls()
    print("Steady-progress integration test passed.")

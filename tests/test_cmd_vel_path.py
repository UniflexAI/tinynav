"""Motion semantics must not depend on a path's world-coordinate origin."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation

from tinynav.platforms.cmd_vel_control import CmdVelControlNode


def controller():
    return SimpleNamespace(
        _nav_active=True, stair_mode=True, _stair_blocked=False, pose=object(),
        T_robot_to_camera=np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                                    [1., 0., 0., 0.], [0., 0., 0., 1.]]),
        lookahead_steps=1, planner_dt=.1, path_pose_stride=10,
        max_forward_speed=1., max_angular_speed=.75, fixed_reverse_speed=.3,
        force_turn_heading_threshold=np.deg2rad(80), last_path_update_time=None,
        path_period_ema=.12, latest_cmd=Twist(), logger=Mock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            to_msg=lambda: SimpleNamespace(sec=0, nanosec=0))))


def path_message(origin=(2.466069528879536, 1.2062086298122559, -8.729683933819434),
                 yaw=.71, pitch=.32, translation=(0., 0., 0.), turn=.107142857142857):
    body = Rotation.from_euler('zy', [yaw, pitch]).as_matrix()
    optical = body @ np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    msg = Path()
    for i in range(2):
        p = PoseStamped()
        xyz = np.array(origin) + i * (body @ np.array(translation))
        p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float, xyz)
        q = Rotation.from_matrix(body @ Rotation.from_euler('z', i * turn).as_matrix()
                                 @ (body.T @ optical)).as_quat()
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = map(float, q)
        msg.poses.append(p)
    return msg


def command(msg):
    node = controller()
    CmdVelControlNode.path_callback(node, msg)
    return node.latest_cmd.linear.x, node.latest_cmd.angular.z


@pytest.mark.parametrize('origin', [(0., 0., 0.), (2.466069528879536, 1.2062086298122559, -8.729683933819434),
                                    (1000., -2000., 300.)])
@pytest.mark.parametrize('turn', [-.75, -.107142857142857, 0., .107142857142857, .75])
def test_stationary_turn_preserves_rotation_at_any_world_origin(origin, turn):
    vx, wz = command(path_message(origin=origin, turn=turn))
    assert vx == 0.
    assert wz == pytest.approx(turn, abs=1e-12)


@pytest.mark.parametrize('translation', [(-1e-10, 0., 0.), (1e-10, 1e-10, 0.), (0., -.0005, 0.)])
def test_submillimeter_position_noise_does_not_choose_bearing_or_reverse(translation):
    assert command(path_message(translation=translation, turn=.2)) == pytest.approx((0., .2))


@pytest.mark.parametrize('translation,expected', [((.25, 0., 0.), (.25, 0.)),
                                                  ((-.3, 0., 0.), (-.3, 0.)),
                                                  ((0., .6, 0.), (0., .75))])
def test_real_motion_retains_forward_reverse_and_rotate_first(translation, expected):
    assert command(path_message(translation=translation, turn=0.)) == pytest.approx(expected)


def test_world_translation_does_not_change_moving_commands():
    rng = np.random.default_rng(23)
    for _ in range(40):
        delta = tuple(rng.uniform(-.5, .5, 3))
        turn = rng.uniform(-.75, .75)
        a = command(path_message(origin=(0., 0., 0.), translation=delta, turn=turn))
        b = command(path_message(origin=(200., -300., -90.), translation=delta, turn=turn))
        assert b == pytest.approx(a, abs=1e-10)


@pytest.mark.parametrize('guard', ['_nav_active', '_stair_blocked'])
def test_control_gate_still_rejects_paths(guard):
    node = controller()
    setattr(node, guard, guard == '_stair_blocked')
    CmdVelControlNode.path_callback(node, path_message(translation=(.3, 0., 0.)))
    assert node.latest_cmd.linear.x == node.latest_cmd.angular.z == 0.
    assert node.last_path_update_time is None

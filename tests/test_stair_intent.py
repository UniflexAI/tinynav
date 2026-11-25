from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import rclpy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String

from tinynav.core import stair_node, planning_node
from tinynav.core.stair_target import StairConfig, StairTargetGenerator
from tinynav.platforms import cmd_vel_control

OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def path(vx=0., turn=0.):
    result = Path()
    for i in range(2):
        p = PoseStamped()
        p.pose.position.x = float(i * vx)
        p.pose.position.z = .3
        q = Rotation.from_matrix(Rotation.from_euler('z', i * turn).as_matrix() @ OPTICAL).as_quat()
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = map(float, q)
        result.poses.append(p)
    return result


@pytest.fixture
def control(monkeypatch):
    rclpy.init()
    clock = SimpleNamespace(monotonic=lambda: 10.)
    monkeypatch.setattr(cmd_vel_control, 'time', clock)
    monkeypatch.setattr(stair_node, 'time', clock)
    n = cmd_vel_control.CmdVelControlNode()
    n.cmd_pub = Mock()
    n.pose = object()
    n._nav_active = True
    yield n, clock
    n.destroy_node()
    rclpy.shutdown()


def test_turn_intent_reaches_control_and_clears_stale_visual_path(control):
    c, clock = control
    args = SimpleNamespace(camera_height=.3, direction='down', floors=1, landings_per_floor=2,
                           turn='left', memory=None, rate=2.)
    n = stair_node.StairNode(args)
    try:
        n.target_pub = Mock(); n.status_pub = Mock(); n.path_pub = Mock()
        n.latest_T = np.eye(4); n.latest_T[:3, :3] = OPTICAL; n.latest_T[2, 3] = .3
        n.last_odom_rx = n.last_depth_rx = 10.
        n.gen.compute = Mock(return_value=dict(status='search', target=np.array([.08, .29, 0.]),
                                               path=None, turn_in_place=True))
        n.timer_callback()
        status = n.status_pub.publish.call_args.args[0]
        assert 'intent=turn' in status.data
        assert not n.path_pub.publish.call_args.args[0].poses
        c.latest_cmd.linear.x = c.prev_cmd.linear.x = .5
        c._on_stair_status(status)
        assert c.cmd_pub.publish.call_args.args[0].linear.x == 0.
        c.path_callback(path(.25, .1))
        c.cmd_timer_callback()
        assert c.cmd_pub.publish.call_args.args[0].linear.x == 0.
        assert c.cmd_pub.publish.call_args.args[0].angular.z == 0.
        c.path_callback(path(0., .3))
        c.cmd_timer_callback()
        assert c.cmd_pub.publish.call_args.args[0].linear.x == 0.
        assert c.cmd_pub.publish.call_args.args[0].angular.z == pytest.approx(.3)
        c._on_stair_status(String(data='down ok intent=move'))
        c.cmd_timer_callback()
        assert c.cmd_pub.publish.call_args.args[0].angular.z == 0.
    finally:
        n.destroy_node()


@pytest.mark.parametrize('vx', [-.3, .25, 1.])
def test_turn_intent_rejects_both_forward_and_reverse_paths(control, vx):
    c, _ = control
    c._on_stair_status(String(data='up search intent=turn'))
    c.path_callback(path(vx, .4))
    c.cmd_timer_callback()
    command = c.cmd_pub.publish.call_args.args[0]
    assert command.linear.x == command.angular.z == 0.
    # A stale moving command cannot bypass the timer's final zero-translation constraint.
    c.latest_cmd.linear.x = vx
    c.latest_cmd.angular.z = .4
    c.last_path_update_time = 10.
    c.prev_cmd.linear.x = vx
    c.cmd_timer_callback()
    assert c.cmd_pub.publish.call_args.args[0].linear.x == 0.


def test_planner_turn_library_contains_no_translation():
    q = Rotation.from_matrix(OPTICAL).as_quat()
    trajectory, params = planning_node.generate_trajectories(np.zeros(3), q, min_forward_speed=.25,
        min_turn_speed=.25, turn_in_place=True)
    assert np.all(params[:, 0] == 0.)
    assert np.allclose(trajectory[:, :, :3], 0.)
    assert np.any(params[:, 1] >= .25) and np.any(params[:, 1] <= -.25)


def test_front_projection_does_not_replace_behind_route_with_high_wall():
    g = StairTargetGenerator(StairConfig(camera_height=.3, max_move_bearing=25.))
    g.direction_sign = -1.; g.turn_side = 1
    T = np.eye(4); T[:3, :3] = OPTICAL; T[2, 3] = .3
    height = np.full((20, 20), .525)
    xy = (np.indices(height.shape).transpose(1, 2, 0) + .5) * .05 - .5
    rel = xy - T[:2, 3]
    available = np.ones_like(height, dtype=bool)
    out = {}
    target = g._keep_in_front(np.array([-.6, .05, -.15]), T[:3, 3], np.array([1., 0.]),
        available, available, np.ones_like(height), xy, height, rel, np.linalg.norm(rel, axis=-1), out)
    assert out['turn_in_place']
    assert target[2] == pytest.approx(0.)
    out = {}
    target = g._keep_in_front(np.array([.4, 0., .525]), T[:3, 3], np.array([1., 0.]),
        available, available, np.ones_like(height), xy, height, rel, np.linalg.norm(rel, axis=-1), out)
    assert target is None and out['status'] == 'no_seed'


@pytest.mark.parametrize('direction,sign', [('up', 1), ('down', -1)])
def test_corrected_moving_target_is_on_its_published_connected_route(direction, sign):
    cfg = StairConfig(camera_height=.4669, auto_camera_height=False)
    g = StairTargetGenerator(cfg)
    xx, yy = np.meshgrid(np.arange(-1., 2.4, .025), np.arange(-.6, .61, .025))
    zz = np.where(xx < 0., 0., sign * .16 * (np.floor(xx / .3) + 1))
    g.frames.append((0., np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])))
    g.latest_stamp = 0.
    T = np.eye(4); T[:3, :3] = OPTICAL; T[:3, 3] = [-.4, 0., cfg.camera_height]
    r = g.compute(T, direction)
    assert r['target'] is not None and not r.get('turn_in_place')
    assert np.min(np.linalg.norm(r['path'] - r['target'], axis=1)) < 1e-8
    assert np.linalg.norm(r['path'][0, :2] - T[:2, 3]) < .08
    if direction == 'down':
        assert np.all(r['height'][r['reachable']] <= cfg.max_step + 1e-8)


def test_visible_route_still_requires_alignment_before_translation():
    g = StairTargetGenerator(StairConfig(camera_height=.3, max_move_bearing=25.))
    cam = np.array([0., 0., .3])
    height = np.zeros((20, 20))
    xy = (np.indices(height.shape).transpose(1, 2, 0) + .5) * .05 - .5
    available = np.ones_like(height, dtype=bool)
    out = {}
    target = g._keep_in_front(np.array([.8, .6, 0.]), cam, np.array([1., 0.]), available,
        available, np.ones_like(height), xy, height, xy, np.linalg.norm(xy, axis=-1), out)
    assert out['turn_in_place']
    assert np.linalg.norm(target[:2]) == pytest.approx(.3)


@pytest.mark.parametrize('bearing', [-12., 12.])
def test_small_platform_alignment_selects_a_turn_without_overshooting(bearing):
    q = Rotation.from_matrix(OPTICAL).as_quat()
    trajectories, params = planning_node.generate_trajectories(np.zeros(3), q,
        min_forward_speed=.25, min_turn_speed=.25, duration=1.5, turn_in_place=True)
    target = np.array([.3 * np.cos(np.deg2rad(bearing)), .3 * np.sin(np.deg2rad(bearing)), 0.])
    node = SimpleNamespace(in_stair_mode=True, _stair_searching=True, _stair_turning=True,
                           last_param=np.zeros(2))
    cost = [planning_node.PlanningNode.trajectory_cost(node, t, p, 0., target, False)
            for t, p in zip(trajectories, params)]
    chosen = params[np.argmin(cost)]
    assert chosen[0] == 0.
    selected = trajectories[np.argmin(cost)]
    forward = Rotation.from_quat(selected[5, 3:]).apply([0., 0., 1.])
    assert np.arctan2(forward[1], forward[0]) * bearing > 0.
    assert abs(chosen[1]) >= .25


@pytest.mark.parametrize('yaw', [-90., 0., 180.])
@pytest.mark.parametrize('pitch', [-20., 0., 20.])
def test_stair_progress_uses_the_same_control_center_as_planning(yaw, pitch):
    robot = planning_node.ROBOT_CONFIG
    cfg = StairConfig(camera_forward_offset=robot.camera_x - robot.control_x,
                      camera_lateral_offset=robot.camera_y - robot.control_y)
    g = StairTargetGenerator(cfg)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('zy', [yaw, pitch], degrees=True).as_matrix() @ OPTICAL
    T[:3, 3] = [2., 1., .6]
    expected = planning_node.PlanningNode.camera_to_robot_center(None, T)
    assert np.allclose(g._control_xy(T), expected[:2])


def test_landing_trajectories_hold_at_goal_without_changing_turns_or_reverse():
    q = Rotation.from_matrix(OPTICAL).as_quat()
    trajectories, params = planning_node.generate_trajectories(np.zeros(3), q,
        min_forward_speed=.25, min_turn_speed=.25, duration=1.5)
    original = trajectories.copy()
    stopped = planning_node.stop_at_target_distance(trajectories, params, np.zeros(3), .2)
    assert np.array_equal(stopped[params[:, 0] <= 0.], original[params[:, 0] <= 0.])
    straight = np.flatnonzero((params[:, 0] == .25) & (params[:, 1] == 0.))[0]
    assert original[straight, -1, 0] > .35
    assert .2 <= stopped[straight, -1, 0] <= .225
    assert np.allclose(stopped[straight, -4:], stopped[straight, -1])
    assert np.array_equal(stopped[straight, :5], original[straight, :5])


@pytest.mark.parametrize('front_support', [True, False])
def test_unfinished_flight_cannot_start_a_landing_uturn(front_support):
    g = StairTargetGenerator(StairConfig(camera_height=.4669, auto_camera_height=False))
    g.was_on_flight = True
    g.pitch = 10.
    g.flight_dir = np.array([1., 0.])
    g.flight_end = np.array([-.1, 0.])
    g.turn_side = 1
    g.latest_stamp = 0.
    xx, yy = np.meshgrid(np.arange(-.3, 1.6, .025),
                         np.arange(-.5, .5, .025) if front_support else np.arange(.8, 1.5, .025))
    g.frames.append((0., np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])))
    T = np.eye(4); T[:3, :3] = OPTICAL; T[2, 3] = .4669
    result = g.compute(T, 'up')
    assert not result.get('uturn')
    if front_support:
        assert result.get('stepping_on') and result['path'][-1, 0] > .4
        assert abs(result['path'][-1, 1]) < .2
        if result.get('turn_in_place'):
            angle = np.degrees(np.arctan2(result['target'][1], result['target'][0]))
            assert abs(angle) < 25.
    else:
        assert result['target'] is None and result['status'] == 'no_seed'


def test_stair_tracking_cost_does_not_treat_camera_height_as_remaining_travel():
    node = SimpleNamespace(in_stair_mode=True, _stair_searching=True, _stair_turning=False,
                           last_param=np.zeros(2))
    q = Rotation.from_matrix(OPTICAL).as_quat()
    trajectory = np.tile(np.r_[.2, 0., .47, q], (16, 1))
    param = np.array([.25, 0.])
    target = np.array([.21, 0., 0.])
    score = planning_node.PlanningNode.trajectory_cost(node, trajectory, param, 0., target, False)
    target[2] = .47
    assert planning_node.PlanningNode.trajectory_cost(node, trajectory, param, 0., target, False) == pytest.approx(score)
    node.in_stair_mode = False
    target[2] = 0.
    assert planning_node.PlanningNode.trajectory_cost(node, trajectory, param, 0., target, False) > score + 400.


def test_crossing_line_stays_fixed_after_the_end_wall_leaves_view():
    g = StairTargetGenerator(StairConfig(camera_height=.4669, auto_camera_height=False))
    g.flight_done = True
    g.flight_dir = np.array([1., 0.])
    g.flight_end = np.array([2.8, 0.])
    g.turn_side = 1
    g.latest_stamp = 1.
    xx, yy = np.meshgrid(np.arange(2.4, 3.6, .025), np.arange(-.6, 1.94, .025))
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    wy, wz = np.meshgrid(np.arange(-.6, 1.94, .025), np.arange(0., .71, .025))
    wall = np.column_stack([np.full(wy.size, 3.6), wy.ravel(), wz.ravel()])
    g.frames.append((1., np.vstack([floor, wall])))
    T = np.eye(4); T[:3, :3] = OPTICAL; T[:3, 3] = [3.1, 0., .4669]
    g.compute(T, 'up')
    assert g.uturn_step_complete and g.uturn_step_distance < .45
    saved = g.uturn_step_distance
    g.frames.clear(); g.frames.append((2., floor)); g.latest_stamp = 2.
    T[:3, :3] = Rotation.from_euler('z', 90., degrees=True).as_matrix() @ OPTICAL
    T[1, 3] = .5
    result = g.compute(T, 'up')
    assert g.uturn_step_distance == saved
    assert result.get('uturn') and not result.get('stepping_in')
    assert result['goal'][0] < 3.25


def test_final_landing_rocking_does_not_move_the_arrival_reference():
    g = StairTargetGenerator(); g.finishing = True
    g.flight_end = np.array([-.4, 1.3]); g.flight_done = True
    expected = g.flight_end.copy()
    T = np.eye(4); T[:3, :3] = Rotation.from_euler('y', -25., degrees=True).as_matrix() @ OPTICAL
    T[:3, 3] = [-.7, 1.3, .5]
    for stamp in np.arange(0., 3., .1):
        g._update_pitch(stamp, T)
    assert np.array_equal(g.flight_end, expected) and g.flight_done


@pytest.mark.parametrize('x,progress,finished', [(-.65, .32, True), (-.35, .32, False), (-.65, .22, False)])
def test_final_arrival_respects_observed_end_wall_and_minimum_step_in(x, progress, finished):
    cfg = StairConfig(camera_height=.4669, auto_camera_height=False)
    g = StairTargetGenerator(cfg); g.latest_stamp = 0.
    g.flight_dir = np.array([-1., 0.]); g.flight_end = np.array([x + progress, 0.])
    xx, yy = np.meshgrid(np.arange(-1.2, .31, .025), np.arange(-.6, .61, .025))
    zz = np.where(xx < 0., 0., .16)
    floor = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    wy, wz = np.meshgrid(np.arange(-.6, .61, .025), np.arange(0., .71, .025))
    wall = np.column_stack([np.full(wy.size, -1.2), wy.ravel(), wz.ravel()])
    g.frames.append((0., np.vstack([floor, wall])))
    T = np.eye(4); T[:3, :3] = Rotation.from_euler('z', 180., degrees=True).as_matrix() @ OPTICAL
    T[:3, 3] = [x, 0., cfg.camera_height]
    result = g.compute(T, 'down', finish_landing=True)
    assert bool(result.get('finished')) == finished

"""Finite stair tasks, input loss, and target recovery without a robot."""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import rclpy
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool, String

from tinynav.core import stair_node
from tinynav.core.stair_node import StairNode
from tinynav.core.stair_target import StairConfig, StairTargetGenerator
from tinynav.platforms.cmd_vel_control import CmdVelControlNode


def camera_pose(z=0.4669, pitch=0.0):
    T = np.eye(4)
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    T[:3, :3] = Rotation.from_euler('y', -pitch, degrees=True).as_matrix() @ optical
    T[2, 3] = z
    return T


@pytest.fixture
def node(monkeypatch):
    rclpy.init()
    args = SimpleNamespace(camera_height=0.4669, direction='down', floors=1,
                           landings_per_floor=2, turn='auto', memory=None, rate=2.0)
    n = StairNode(args)
    n.target_pub = Mock()
    n.poi_change_pub = Mock()
    n.status_pub = Mock()
    n.last_depth_rx = n.last_odom_rx = 10.0
    n.latest_T = camera_pose()
    n.stopped = False
    monkeypatch.setattr(stair_node, 'time', SimpleNamespace(monotonic=lambda: 10.0))
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_one_floor_stops_after_two_landings_and_latches(node):
    node.gen.landings = 1
    node.gen.compute = Mock(return_value={'target': np.array([1., 0., -1.44]), 'path': None, 'status': 'search'})
    node.timer_callback()
    assert node.target_pub.publish.call_count == 1
    assert 'floors=0/1' in node.status_pub.publish.call_args.args[0].data

    node.gen.landings = 2
    node.gen.compute.return_value = {'status': 'finished', 'finished': True}
    node.timer_callback()
    assert node.arrived
    assert node.target_pub.publish.call_count == 1
    assert 'floors=1/1' in node.status_pub.publish.call_args.args[0].data
    assert 'arrived' in node.status_pub.publish.call_args.args[0].data
    node.gen.landings = 3
    node.timer_callback()
    assert node.target_pub.publish.call_count == 1
    assert node.gen.compute.call_count == 2


def test_depth_timeout_stops_even_while_odometry_is_fresh(node):
    node.last_depth_rx = 9.0
    node.timer_callback()
    assert 'depth_invalid' in node.status_pub.publish.call_args.args[0].data
    node.poi_change_pub.publish.assert_called_once()
    node.target_pub.publish.assert_not_called()


def test_pause_and_resume_preserve_task_count(node):
    node.gen.landings = 1
    node.gen.compute = Mock(return_value={'target': np.array([1., 0., -1.44]), 'path': None, 'status': 'search'})
    node.pause_callback(Bool(data=True))
    node.timer_callback()
    node.target_pub.publish.assert_not_called()
    node.pause_callback(Bool(data=False))
    node.timer_callback()
    assert node.gen.landings == 1
    node.target_pub.publish.assert_called_once()


def test_new_task_clears_old_count_and_latched_arrival(node):
    node.arrived = True
    node.gen.landings = 2
    node.cmd_callback(String(data='down left'))
    assert not node.arrived and node.gen.landings == 0
    assert node.gen.turn_side == 1 and node.landings_goal == 2


@pytest.mark.parametrize('direction,sign', [('down', -1), ('up', 1)])
def test_pitch_cycle_counts_each_landing_once(direction, sign):
    gen = StairTargetGenerator()
    gen.direction_sign = sign
    gen.landing_ref_z = 0.4669
    stamp = 0.0
    for flight in range(2):
        for j in range(32):
            stamp += 0.1
            z = 0.4669 + sign * 1.44 * (flight + (j + 1) / 32)
            gen.add_pose(stamp, (stamp * .15, 0., z))
            gen._update_pitch(stamp, camera_pose(z, sign * 20))
        for j in range(35):
            stamp += 0.1
            z = 0.4669 + sign * 1.44 * (flight + 1)
            gen.add_pose(stamp, (stamp * .15, 0., z))
            gen._update_pitch(stamp, camera_pose(z))
        assert gen.landings == flight + 1


def test_jump_does_not_count_as_floor_progress():
    gen = StairTargetGenerator()
    gen.landing_ref_z = 0.4669
    gen.add_pose(1.0, (0., 0., 0.4669))
    gen.add_pose(1.01, (0., 0., -5.5331))
    assert not gen.odom_valid and gen.landings == 0
    assert gen.landing_ref_z == pytest.approx(-5.5331)


def test_stalled_landing_target_recovery_preserves_flight_and_count():
    gen = StairTargetGenerator()
    gen.last_status = 'search'
    gen.flight_dir = np.array([1., 0.])
    gen.landings = 1
    gen.search_goal = (np.array([1., 1.]), 0.0)
    gen.latest_stamp = 0.0
    gen._check_search_progress(np.zeros(3), np.array([1., 0.]))
    gen.latest_stamp = 6.1
    gen._check_search_progress(np.zeros(3), np.array([1., 0.]))
    assert gen.search_goal is None and gen.search_recovery_until > gen.latest_stamp
    np.testing.assert_array_equal(gen.stalled_goal, [1., 1.])
    np.testing.assert_array_equal(gen.flight_dir, [1., 0.])
    assert gen.landings == 1


def test_turning_counts_as_search_progress():
    gen = StairTargetGenerator()
    gen.last_status = 'search'
    gen.latest_stamp = 0.
    gen._check_search_progress(np.zeros(3), np.array([1., 0.]))
    gen.latest_stamp = 6.1
    gen._check_search_progress(np.zeros(3), np.array([0., 1.]))
    assert gen.search_recovery_until == -np.inf


def test_stair_stop_status_immediately_zeros_control():
    rclpy.init()
    node = CmdVelControlNode()
    try:
        node.cmd_pub = Mock()
        node._nav_active = True
        node.latest_cmd.linear.x = 0.3
        node._on_stair_status(String(data='down depth_invalid landings=1/2'))
        assert node.cmd_pub.publish.call_args.args[0].linear.x == 0.0
        node.cmd_timer_callback()
        assert node.cmd_pub.publish.call_args.args[0].linear.x == 0.0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_api_passes_run_specific_floors(monkeypatch):
    from app.backend.routers import nav
    fake = Mock(state='idle')
    fake.cmd_stair_start.return_value = {'floors': 1, 'landings_per_floor': 2}
    monkeypatch.setattr(nav.runner, 'node', fake)
    result = nav.nav_stair_start(nav.StairRequest(direction='down', floors=1))
    fake.cmd_stair_start.assert_called_once_with('down', 'auto', floors=1)
    assert result['floors'] == 1 and result['landings_per_floor'] == 2


def test_launch_uses_run_count_without_changing_saved_settings():
    from app.backend.node_manager import BackendNode
    fake = Mock()
    settings = {'floors': 4, 'landings_per_floor': 2, 'camera_height': .4669}
    fake.stair_settings.side_effect = lambda: settings.copy()
    fake._lock = __import__('threading').Lock()
    fake._cmd_vel_proc = None
    result = BackendNode.cmd_stair_start(fake, 'down', floors=1)
    cmd = fake._launch_proc.call_args.args[1]
    assert cmd[cmd.index('--floors') + 1] == '1'
    assert cmd[cmd.index('--landings_per_floor') + 1] == '2'
    assert result['floors'] == 1
    assert fake.stair_settings() == settings


@pytest.mark.parametrize('fields', [
    {'floors': -1}, {'floors': 51}, {'landings_per_floor': 0}, {'landings_per_floor': 5},
])
def test_api_rejects_invalid_stair_counts(fields):
    from app.backend.routers.nav import StairRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StairRequest(direction='down', **fields)


def test_start_route_roundtrip_and_legacy_settings(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.backend.routers import nav
    fake = Mock(state='idle')
    fake.cmd_stair_start.return_value = {'floors': 1, 'landings_per_floor': 2}
    monkeypatch.setattr(nav.runner, 'node', fake)
    app = FastAPI()
    app.include_router(nav.router, prefix='/nav')
    with TestClient(app) as client:
        response = client.post('/nav/stair/start', json={'direction': 'down', 'floors': 1, 'landings_per_floor': 2})
        assert response.status_code == 200 and response.json()['floors'] == 1
        fake.cmd_stair_start.assert_called_with('down', 'auto', floors=1, landings_per_floor=2)
        response = client.post('/nav/stair/start', json={'direction': 'down'})
        assert response.status_code == 200
        fake.cmd_stair_start.assert_called_with('down', 'auto')
        response = client.post('/nav/stair/start', json={'direction': 'down', 'floors': -1})
        assert response.status_code == 422


def test_arrival_result_survives_stair_process_cleanup():
    from app.backend.node_manager import BackendNode
    fake = Mock()
    fake._lock = __import__('threading').Lock()
    fake._stair_mode = 'down'
    fake._stair_status = 'down arrived landings=2/2 floors=1/1'
    BackendNode._kill_stair_node(fake, keep_status=True)
    assert fake._stair_mode is None and 'floors=1/1' in fake._stair_status
    BackendNode._kill_stair_node(fake)
    assert fake._stair_status == ''


def test_stationary_search_replans_on_the_observed_landing():
    cfg = StairConfig(resolution=.1, half_size=2., auto_camera_height=False)
    gen = StairTargetGenerator(cfg)
    xx, yy = np.meshgrid(np.arange(-.9, 1.5, .05), np.arange(-.7, 1.5, .05))
    corridor = (abs(yy) < .6) | ((xx > .5) & (xx < 1.4))
    points = np.column_stack((xx[corridor], yy[corridor], np.zeros(corridor.sum())))
    T = camera_pose(cfg.camera_height)
    gen.flight_dir = np.array([1., 0.])
    gen.well_side = 5.
    results = []
    for stamp in np.arange(0., 7., .5):
        gen.latest_stamp = stamp
        gen.frames.clear()
        gen.frames.append((stamp, points))
        results.append(gen.compute(T, 'down'))
    recovering = [res for res in results if res.get('search_recovery')]
    assert recovering
    for res in recovering:
        assert res['target'] is not None
        if res['path'] is not None:
            index = np.floor((res['path'][:, :2] - res['origin']) / cfg.resolution).astype(int)
            assert res['free'][index[:, 0], index[:, 1]].all()
        assert gen.landings == 0


@pytest.mark.parametrize('direction,sign', [('down', -1), ('up', 1)])
@pytest.mark.parametrize('heading', [-11., 11.])
def test_measured_treads_align_target_before_entering_flight(direction, sign, heading):
    cfg = StairConfig(camera_height=.4669, auto_camera_height=False)
    gen = StairTargetGenerator(cfg)
    xx, yy = np.meshgrid(np.arange(-1., 2.4, .025), np.arange(-.6, .61, .025))
    zz = np.where(xx < 0., 0., sign * .16 * (np.floor(xx / .3) + 1))
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    T = camera_pose(cfg.camera_height)
    T[:3, :3] = Rotation.from_euler('z', heading, degrees=True).as_matrix() @ T[:3, :3]
    T[0, 3] = -.4
    gen.latest_stamp = 0.
    gen.frames.append((0., points))

    result = gen.compute(T, direction)

    assert not gen.on_flight and gen.flight_dir is None
    assert result['status'] == 'ok'
    assert result.get('tread_aligned')
    assert result['target'][0] > .3
    assert abs(result['target'][1]) < .1
    cell = np.floor((result['target'][:2] - result['origin']) / cfg.resolution).astype(int)
    assert result['reachable'][tuple(cell)]
    assert np.isfinite(result['height'][tuple(cell)])


def test_two_floors_continues_through_first_floor_and_stops_at_four(node):
    node.floors, node.landings_goal = 2, 4
    node.gen.compute = Mock(return_value={'target': np.array([1., 0., -1.44]), 'path': None, 'status': 'search'})
    for landing in range(4):
        node.gen.landings = landing
        node.timer_callback()
        assert not node.arrived
        assert f'floors={landing // 2}/2' in node.status_pub.publish.call_args.args[0].data
    assert node.target_pub.publish.call_count == 4
    node.gen.landings = 4
    node.gen.compute.return_value = {'status': 'finished', 'finished': True}
    node.timer_callback()
    assert node.arrived
    assert 'landings=4/4 floors=2/2' in node.status_pub.publish.call_args.args[0].data
    for _ in range(5):
        node.timer_callback()
    assert node.target_pub.publish.call_count == 4


def test_final_count_still_steps_onto_observed_flat_landing(node):
    node.gen.landings = node.landings_goal
    node.gen.latest_stamp = 0.
    node.gen.flight_dir = np.array([1., 0.])
    node.gen.flight_end = np.array([-.2, 0.])
    xx, yy = np.meshgrid(np.arange(-.3, 1., .025), np.arange(-.6, .61, .025))
    node.gen.frames.append((0., np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))))

    node.timer_callback()

    assert not node.arrived
    target = node.target_pub.publish.call_args.args[0].pose.pose.position
    assert target.x > .1 and abs(target.y) < .1
    assert 'finishing' in node.status_pub.publish.call_args.args[0].data
    node.latest_T[0, 3] = .3
    node.timer_callback()
    assert node.arrived
    assert 'arrived' in node.status_pub.publish.call_args.args[0].data


def test_final_count_without_measured_support_stops_without_claiming_arrival(node):
    node.gen.landings = node.landings_goal
    node.gen.latest_stamp = 0.
    node.gen.flight_dir = np.array([1., 0.])
    node.gen.flight_end = np.array([-.2, 0.])
    node.timer_callback()
    assert not node.arrived
    node.target_pub.publish.assert_not_called()
    assert 'no_seed' in node.status_pub.publish.call_args.args[0].data


def test_final_landing_does_not_request_backtracking_when_facing_away(node):
    node.gen.landings = node.landings_goal
    node.gen.latest_stamp = 0.
    node.gen.flight_dir = np.array([1., 0.])
    node.gen.flight_end = np.array([-.2, 0.])
    node.latest_T[:3, :3] = Rotation.from_euler('z', 180., degrees=True).as_matrix() @ node.latest_T[:3, :3]
    xx, yy = np.meshgrid(np.arange(-.3, 1., .025), np.arange(-.6, .61, .025))
    node.gen.frames.append((0., np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))))
    node.timer_callback()
    assert not node.arrived
    node.target_pub.publish.assert_not_called()
    assert 'no_seed' in node.status_pub.publish.call_args.args[0].data


@pytest.mark.parametrize('height', [0., -.4])
def test_final_landing_uses_offset_measured_support_but_not_a_drop(node, height):
    node.gen.landings = node.landings_goal
    node.gen.latest_stamp = 0.
    node.gen.flight_dir = np.array([1., 0.])
    node.gen.flight_end = np.array([-.2, 0.])
    xx, yy = np.meshgrid(np.arange(.1, 1., .025), np.arange(.425, .9, .025))
    node.gen.frames.append((0., np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, height)))))
    node.timer_callback()
    assert not node.arrived
    if height == 0.:
        target = node.target_pub.publish.call_args.args[0].pose.pose.position
        assert target.y > node.gen.cfg.search_clearance
        assert target.x > .1
        assert 'finishing' in node.status_pub.publish.call_args.args[0].data
    else:
        node.target_pub.publish.assert_not_called()
        assert 'no_seed' in node.status_pub.publish.call_args.args[0].data


def test_final_count_without_seeds_does_not_start_another_turn(node):
    node.gen.landings = node.landings_goal
    node.gen.latest_stamp = 0.
    node.gen.flight_dir = np.array([1., 0.])
    node.gen.flight_end = np.array([-.2, 0.])
    node.gen.well_side = 6.
    node.timer_callback()
    assert not node.arrived
    node.target_pub.publish.assert_not_called()
    assert 'no_seed' in node.status_pub.publish.call_args.args[0].data


@pytest.mark.parametrize('enabled', ['0', 'false', 'no'])
def test_deployment_can_disable_unitree_backend(monkeypatch, enabled):
    from app.backend.node_manager import BackendNode
    monkeypatch.setenv('TINYNAV_UNITREE_ENABLED', enabled)
    fake = Mock()
    BackendNode._start_unitree_if_configured(fake)
    fake._launch_proc.assert_not_called()


def test_deployment_uses_configured_unitree_interface(monkeypatch):
    from app.backend.node_manager import BackendNode
    monkeypatch.setenv('TINYNAV_UNITREE_ENABLED', '1')
    monkeypatch.setenv('UNITREE_NETWORK_INTERFACE', 'enp3s0')
    fake = Mock()
    BackendNode._start_unitree_if_configured(fake)
    cmd = fake._launch_proc.call_args.args[1]
    assert cmd[-2:] == ['--network-interface', 'enp3s0']


def test_deployment_rejects_unknown_unitree_enable_value(monkeypatch):
    from app.backend.node_manager import BackendNode
    monkeypatch.setenv('TINYNAV_UNITREE_ENABLED', 'typo')
    with pytest.raises(ValueError):
        BackendNode._start_unitree_if_configured(Mock())

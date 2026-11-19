import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from tinynav.core.navigation_safety import NavigationGuard, NavigationLimits, MotionResponse, executable_speed
from tinynav.rl.route_rejoin import RouteRejoin


LIMITS = NavigationLimits(front=.30, rear=.30, half_width=.15)


def context(*, wall_x=None, wall_y=None, speed=0., delay=0.):
    origin = np.array([-5., -5.])
    coordinates = origin[0] + (np.arange(100) + .5) * .10
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    occupied = np.zeros((100, 100), dtype=bool)
    if wall_x is not None:
        occupied |= x >= wall_x
    if wall_y is not None:
        occupied |= y >= wall_y
    esdf = distance_transform_edt(~occupied).astype(np.float32) * .10 if occupied.any() else np.full((100, 100), 9.)
    return NavigationGuard(LIMITS, [0., 0., 0.], (speed, 0.), esdf, origin, .10, delay=delay)


def test_deceleration_is_checked_from_the_previous_base_command():
    guard = context(wall_x=.75, speed=.55, delay=.12)
    valid, room, _ = guard.evaluate([(.10, 0.), (.55, 0.), (0., 0.)])
    assert not valid[1]
    assert valid[2] and room[2] >= .10
    slow = context(wall_x=.75)
    assert slow.evaluate([(.10, 0.)])[0][0]
    assert guard.trajectories([(.10, 0.)])[0, -1, 0] > slow.trajectories([(.10, 0.)])[0, -1, 0]


def test_rotation_uses_the_full_shared_rectangle():
    guard = context(wall_y=.40)
    valid, room, _ = guard.evaluate([(.10, 0.), (0., .75)])
    assert valid[0]
    assert not valid[1] and room[1] < .10


def test_under_clearance_escape_must_improve_room():
    guard = context(wall_x=.35)
    assert 0 < guard.now < .10
    valid, room, paths = guard.evaluate([(-.10, 0.), (.10, 0.)])
    assert valid[0] and not valid[1]
    assert guard.clearance(paths[0, -1:])[0] > guard.now


def test_stale_pose_and_map_boundary_are_rejected():
    assert not context(delay=.60).evaluate([(.10, 0.)])[0][0]
    guard = context()
    guard.pose[0] = 4.8
    assert not guard.evaluate([(.10, 0.)])[0][0]


def test_planning_grids_with_a_three_dimensional_origin_are_supported():
    guard = context()
    copied = NavigationGuard(LIMITS, guard.pose, guard.command, guard.esdf, [-5., -5., -1.55], .10)
    assert copied.evaluate([(.10, 0.)])[0][0]


def test_executable_speed_matches_engagement_and_reverse_transition():
    assert np.isclose(executable_speed(0., .20, LIMITS.period, LIMITS), .10)
    assert np.isclose(executable_speed(0., -.20, LIMITS.period, LIMITS), -.10)
    assert np.isclose(executable_speed(.55, 0., LIMITS.period, LIMITS), .50)
    assert np.isclose(executable_speed(-.20, .20, LIMITS.period, LIMITS), -.15)
    assert not LIMITS.executable([(np.nan, .30), (.20, np.inf)]).any()


def test_shared_limits_use_recorded_geometry_and_reject_invalid_configuration(monkeypatch):
    from tinynav.core.robot_specs import ROBOT_CONFIG
    monkeypatch.setenv("TINYNAV_PPO_FRONT_M", ".30")
    monkeypatch.setenv("TINYNAV_NAV_FRONT_M", ".32")
    monkeypatch.setenv("TINYNAV_NAV_MAX_REVERSE_MPS", "3")
    limits = NavigationLimits.from_environment()
    assert limits.front == .32 and limits.max_reverse == ROBOT_CONFIG.max_linear_vel
    monkeypatch.delenv("TINYNAV_NAV_FRONT_M")
    assert NavigationLimits.from_environment().front == .30
    monkeypatch.setenv("TINYNAV_NAV_MAX_FORWARD_MPS", "nan")
    with pytest.raises(ValueError, match="finite"):
        NavigationLimits.from_environment()


def test_nonresponsive_ordinary_turn_is_detected_before_one_second():
    response = MotionResponse()
    response.issued((0., .30))
    assert not response.stalled(0., np.array([0., 0., 0.]))
    assert response.stalled(.81, np.array([.002, 0., .01]))
    response.issued((.10, 0.))
    assert not response.stalled(1., np.array([0., 0., 0.]))
    assert not response.stalled(1.81, np.array([.08, 0., 0.]))


def test_forward_merge_skips_an_initial_sideways_route_segment():
    route = np.array([[0., -.40], [-.10, -.40], [-.10, -.20], [.20, -.20], [1., -.20], [2., -.20], [3., -.20]])
    guard = context()
    merge = RouteRejoin()
    command = merge.choose(guard, route, np.array([3., -.20]))
    assert command is not None and command[0] > 0
    assert guard.evaluate([command])[0][0]
    assert merge.target[0] > .40 and merge.clearance >= .10
    blocked = context(wall_x=.60)
    assert merge.choose(blocked, route, np.array([3., -.20])) is None
    assert merge.curve is None


def test_library_optical_pitch_is_converted_to_the_correct_body_turn():
    from scipy.spatial.transform import Rotation
    from tinynav.core.planning_node import generate_trajectories, library_commands
    rotation = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    quaternion = Rotation.from_matrix(rotation).as_quat()
    trajectories, parameters = generate_trajectories(np.zeros(3), quaternion, .55, .75)
    index = int(np.argmax(parameters[:, 0] + parameters[:, 1]))
    command = library_commands(parameters)[index]
    assert command[0] > 0 and command[1] < 0
    expected = context().trajectories([command], braking=False)[0, -1]
    assert expected[1] < 0 and trajectories[index, -1, 1] < 0
    assert np.isclose(parameters[:, 0].max(), .55)


def test_ppo_ordinary_command_enters_checked_recovery_on_no_response():
    from tests.test_rl_planner import FakePolicy, _scene, _camera_pose, CAM
    from tinynav.rl.ppo_planner import RLPlanner, PPOGeometry
    grid, origin, esdf = _scene()
    planner = RLPlanner(FakePolicy([.35, 0.]), PPOGeometry(.30, .30, .15))
    for stamp in (0., .10, .91):
        guard = NavigationGuard(LIMITS, [0., 0., 0.], (.35, 0.), esdf, origin, .10, stamp=stamp)
        planner.navigation_guard_factory = lambda guard=guard: guard
        v, w, _, _, status = planner.step(grid, origin, _camera_pose(CAM, 0., 0.),
                                        stamp, np.array([3., 0.]), esdf)
    assert planner.stuck_count == 1
    assert status.startswith("recovery")
    assert guard.evaluate([(v, w)])[0][0]

import json
from dataclasses import replace

import numpy as np
import pytest

from tinynav.core import robot_specs
from tinynav.core.robot_specs import B2_CONFIG, GO2_CONFIG, with_measured_camera


def test_no_file_is_the_type_default(tmp_path):
    assert with_measured_camera(GO2_CONFIG, tmp_path / 'absent.json') is GO2_CONFIG


def test_a_measurement_moves_the_camera_and_nothing_else(tmp_path):
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'forward_m': 0.31, 'left_m': -0.04, 'residual_m': 0.01}))
    got = with_measured_camera(GO2_CONFIG, path)
    assert list(got.cam_offset_3d) == pytest.approx([0.04, 0.0, 0.31])   # left is camera -x
    assert got.footprint_from_control() == GO2_CONFIG.footprint_from_control()
    assert got.name == GO2_CONFIG.name


def test_a_floor_height_puts_the_band_bottom_above_that_floor(tmp_path):
    path = tmp_path / 'camera_offset.json'
    doc = {'forward_m': 0.3, 'left_m': 0.0, 'up_m': 0.6}
    path.write_text(json.dumps(doc))
    assert with_measured_camera(GO2_CONFIG, path).obstacle == GO2_CONFIG.obstacle
    path.write_text(json.dumps({**doc, 'floor_at': '2026-11-04T08:04:32'}))
    got = with_measured_camera(GO2_CONFIG, path).obstacle
    assert got.robot_z_bottom == pytest.approx(-(0.6 - robot_specs.FLOOR_CLEARANCE_M))
    assert replace(got, robot_z_bottom=GO2_CONFIG.obstacle.robot_z_bottom) == GO2_CONFIG.obstacle


@pytest.mark.parametrize('body', ['{"forward_m": 0.3}', 'not json', '{"forward_m": "x", "left_m": 0}'])
def test_an_unreadable_file_is_the_type_default_and_says_so(tmp_path, capsys, body):
    path = tmp_path / 'camera_offset.json'
    path.write_text(body)
    assert with_measured_camera(GO2_CONFIG, path) is GO2_CONFIG
    assert str(path) in capsys.readouterr().err


def test_only_the_free_gait_has_floors_of_its_own(tmp_path):
    missing = tmp_path / 'none.json'
    classic, free = (robot_specs.robot_config(missing, gait) for gait in ('classic', 'free'))
    floors = robot_specs.FREE_GAIT_FLOORS.get(classic.name, {})
    assert replace(classic, **floors) == free


def test_a_camera_to_the_left_has_the_origin_to_its_right():
    """cam_offset_3d is in the camera's frame (x right, y down, z forward), as
    pilot/geometry.camera_from_robot has it: a camera 0.05 m left of the origin puts
    the origin 0.05 m to the camera's right."""
    cfg = replace(GO2_CONFIG, camera_fwd=0.3, camera_left=0.05)
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]   # camera z along world +x
    origin = T[:3, 3] - T[:3, :3] @ cfg.cam_offset_3d
    assert origin[:2] == pytest.approx([-0.3, -0.05])


def test_the_rig_file_sets_the_footprint_by_hand_and_each_part_alone(tmp_path):
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'front_len': 0.44, 'rear_len': 0.44, 'half_width': 0.175,
                                'why': 'a reason'}))
    got = with_measured_camera(B2_CONFIG, path)
    assert (got.front_len, got.rear_len, got.half_width) == (0.44, 0.44, 0.175)
    assert got.camera_fwd == B2_CONFIG.camera_fwd and got.obstacle == B2_CONFIG.obstacle
    path.write_text(json.dumps({'forward_m': 0.3, 'left_m': 0.02, 'front_len': 0.5,
                                'rear_len': 0.5, 'half_width': 0.2}))
    got = with_measured_camera(B2_CONFIG, path)
    assert (got.camera_fwd, got.front_len) == (0.3, 0.5)
    path.write_text('{"front_len": 0.44}')
    assert with_measured_camera(B2_CONFIG, path) is B2_CONFIG


TYPES = {n: c for n, c in vars(robot_specs).items()
         if n.endswith('_CONFIG') and not n.startswith('_') and n != 'ROBOT_CONFIG'}


ONE = (1.0, 1.0, 1.0)


@pytest.mark.parametrize('name', sorted(TYPES))
def test_without_a_scale_the_collision_footprint_is_the_body(name):
    config = TYPES[name]
    assert config.footprint_scale == ONE
    assert config.collision_footprint() == config.footprint_from_control()


def test_a_file_without_a_scale_keeps_it_at_one(tmp_path):
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'forward_m': 0.3, 'left_m': 0.0, 'front_len': 0.4,
                                'rear_len': 0.4, 'half_width': 0.2}))
    got = robot_specs.robot_config(path)
    assert got.footprint_scale == ONE
    assert got.collision_footprint() == got.footprint_from_control() == (0.4, 0.4, 0.2)


def test_a_scale_shrinks_what_is_checked_and_nothing_else(tmp_path):
    """The body stays the body: the arrival radii are measured in it, and the camera,
    the band, safety_radius and the velocity limits are not the scale's."""
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'footprint_scale': 0.8}))
    got = with_measured_camera(GO2_CONFIG, path)
    fl, rl, hw = GO2_CONFIG.footprint_from_control()
    assert got.footprint_scale == (0.8, 0.8, 0.8)
    assert got.collision_footprint() == pytest.approx((0.8 * fl, 0.8 * rl, 0.8 * hw))
    assert got.footprint_from_control() == (fl, rl, hw)
    assert got.arrival_radius == GO2_CONFIG.arrival_radius == 2.0 * hw
    assert got.planned_arrival_radius == GO2_CONFIG.planned_arrival_radius == 1.5 * hw
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    assert got.arrival_distance(T, (1.0, 0.4)) == GO2_CONFIG.arrival_distance(T, (1.0, 0.4))
    assert replace(got, footprint_scale=ONE) == GO2_CONFIG


def test_each_direction_has_its_own_factor(tmp_path):
    """A passage tight on the sides wants the width alone: the nose and the tail keep
    the body's length. A direction the file leaves out is 1.0."""
    path = tmp_path / 'camera_offset.json'
    fl, rl, hw = GO2_CONFIG.footprint_from_control()
    path.write_text(json.dumps({'footprint_scale': {'width': 0.7}}))
    got = with_measured_camera(GO2_CONFIG, path)
    assert got.footprint_scale == (1.0, 1.0, 0.7)
    assert got.collision_footprint() == pytest.approx((fl, rl, 0.7 * hw))
    path.write_text(json.dumps({'footprint_scale': {'front': 0.9, 'rear': 0.6, 'width': 0.8}}))
    got = with_measured_camera(GO2_CONFIG, path)
    assert got.footprint_scale == (0.9, 0.6, 0.8)
    assert got.collision_footprint() == pytest.approx((0.9 * fl, 0.6 * rl, 0.8 * hw))


def test_the_scale_applies_to_a_footprint_set_by_hand(tmp_path):
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'front_len': 0.44, 'rear_len': 0.44, 'half_width': 0.175,
                                'footprint_scale': 0.8}))
    got = with_measured_camera(B2_CONFIG, path)
    assert (got.front_len, got.rear_len, got.half_width) == (0.44, 0.44, 0.175)
    assert got.collision_footprint() == pytest.approx((0.352, 0.352, 0.14))


@pytest.mark.parametrize('value, held', [(0.3, 0.5), (1.4, 1.0), (0.5, 0.5), (1.0, 1.0), (0.75, 0.75)])
def test_a_scale_out_of_range_is_clamped_and_says_so(tmp_path, capsys, value, held):
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'footprint_scale': value}))
    assert with_measured_camera(GO2_CONFIG, path).footprint_scale == (held,) * 3
    assert ('outside' in capsys.readouterr().err) == (value != held)
    path.write_text(json.dumps({'footprint_scale': {'rear': value}}))
    assert with_measured_camera(GO2_CONFIG, path).footprint_scale == (1.0, held, 1.0)


@pytest.mark.parametrize('bad', ['"narrow"', 'NaN', 'null', '{"width": "narrow"}',
                                 '{"sides": 0.8}', '"0.9,0.8"'])
def test_a_scale_that_is_not_a_scale_is_refused(tmp_path, capsys, bad):
    """Not a number, or a direction that does not exist: the file is unreadable, as
    any bad key makes it."""
    path = tmp_path / 'camera_offset.json'
    path.write_text('{"forward_m": 0.3, "left_m": 0.0, "footprint_scale": %s}' % bad)
    assert with_measured_camera(GO2_CONFIG, path) is GO2_CONFIG
    assert str(path) in capsys.readouterr().err


def test_the_g1_ramps_at_half_the_default_acceleration():
    """A humanoid pitches over on the ramp a quadruped takes (8038, 2026-10-21)."""
    from tinynav.core.robot_specs import G1_CONFIG
    assert G1_CONFIG.max_linear_acc == pytest.approx(0.3)


@pytest.mark.parametrize('name', sorted(n for n in TYPES if n != 'G1_CONFIG'))
def test_every_other_model_keeps_the_ramp_it_had(name):
    assert TYPES[name].max_linear_acc == pytest.approx(0.6)


import json
from dataclasses import replace

import numpy as np
import pytest

from tinynav.core import robot_specs
from tinynav.core.robot_specs import GO2_CONFIG, with_measured_camera


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

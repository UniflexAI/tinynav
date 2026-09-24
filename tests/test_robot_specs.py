import json

import pytest

from tinynav.core.robot_specs import GO2_CONFIG, with_measured_camera


def test_no_file_is_the_type_default(tmp_path):
    assert with_measured_camera(GO2_CONFIG, tmp_path / 'absent.json') is GO2_CONFIG


def test_a_measurement_moves_the_camera_and_nothing_else(tmp_path):
    path = tmp_path / 'camera_offset.json'
    path.write_text(json.dumps({'forward_m': 0.31, 'left_m': -0.04, 'residual_m': 0.01}))
    got = with_measured_camera(GO2_CONFIG, path)
    assert list(got.cam_offset_3d) == pytest.approx([-0.04, 0.0, 0.31])
    assert got.footprint_from_control() == GO2_CONFIG.footprint_from_control()
    assert got.name == GO2_CONFIG.name


@pytest.mark.parametrize('body', ['{"forward_m": 0.3}', 'not json', '{"forward_m": "x", "left_m": 0}'])
def test_an_unreadable_file_is_the_type_default_and_says_so(tmp_path, capsys, body):
    path = tmp_path / 'camera_offset.json'
    path.write_text(body)
    assert with_measured_camera(GO2_CONFIG, path) is GO2_CONFIG
    assert str(path) in capsys.readouterr().err

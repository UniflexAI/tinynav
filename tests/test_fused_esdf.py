import ast
from pathlib import Path

import cv2
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt


PLANNING_NODE = Path(__file__).resolve().parents[1] / 'tinynav' / 'core' / 'planning_node.py'


def _load_fused_esdf_symbols():
    source = PLANNING_NODE.read_text()
    module = ast.parse(source)
    wanted = {
        'FusedESDFConfig',
        '_build_step_obstacle_from_height',
        '_build_wall_obstacle_from_occupancy',
        '_build_esdf_from_obstacle',
        'build_fused_esdf_from_height',
    }
    selected = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    mini_module = ast.Module(body=selected, type_ignores=[])
    code = compile(mini_module, str(PLANNING_NODE), 'exec')
    ns = {
        'np': np,
        'cv2': cv2,
        'dataclass': dataclass,
        'distance_transform_edt': distance_transform_edt,
    }
    exec(code, ns)
    return ns


SYMS = _load_fused_esdf_symbols()
FusedESDFConfig = SYMS['FusedESDFConfig']
build_fused_esdf_from_height = SYMS['build_fused_esdf_from_height']


def test_step_and_wall_esdf_are_fused_by_minimum():
    height = np.zeros((7, 7), dtype=np.float32)
    height[:, 4:] = 0.5  # step discontinuity at col 3/4

    occupancy = np.zeros((7, 7, 5), dtype=np.float32)
    occupancy[1, 1, 2] = 1.0  # wall obstacle away from the step

    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    config = FusedESDFConfig(
        max_step_height=0.3,
        occ_threshold=0.1,
        robot_z_bottom=-0.2,
        robot_z_top=3.0,
        step_denoise_kernel=1,
    )

    fused_esdf, slope, unknown_mask, debug = build_fused_esdf_from_height(
        height, occupancy, origin, 1.0, 0.0, config=config, return_debug=True
    )

    np.testing.assert_allclose(fused_esdf, np.minimum(debug['step_esdf'], debug['wall_esdf']))
    assert debug['step_obstacle_mask'].any()
    assert debug['wall_obstacle_mask'].any()
    assert slope[3, 3] > 0.0
    assert not unknown_mask.any()


def test_unknown_mask_only_tracks_invalid_height_without_wall_override():
    height = np.zeros((5, 5), dtype=np.float32)
    height[2, 2] = np.nan

    occupancy = np.zeros((5, 5, 4), dtype=np.float32)
    occupancy[2, 2, 1] = 1.0  # wall at the invalid-height cell should suppress unknown there

    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cfg = FusedESDFConfig(robot_z_bottom=-0.2, robot_z_top=2.0)
    fused_esdf, slope, unknown_mask, debug = build_fused_esdf_from_height(
        height, occupancy, origin, 1.0, 0.0, config=cfg, return_debug=True
    )

    assert unknown_mask.sum() == 0
    assert debug['wall_obstacle_mask'][2, 2]
    assert fused_esdf.shape == height.shape
    assert slope.shape == height.shape


def test_config_controls_wall_height_band_and_step_denoise_kernel():
    height = np.zeros((5, 5), dtype=np.float32)
    height[2, 2] = 1.0  # isolated spike

    occupancy = np.zeros((5, 5, 4), dtype=np.float32)
    occupancy[4, 4, 3] = 1.0  # high voxel, should only appear with expanded z band
    origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    tight_cfg = FusedESDFConfig(step_denoise_kernel=3, robot_z_bottom=-0.2, robot_z_top=0.8)
    _, _, _, tight_debug = build_fused_esdf_from_height(
        height, occupancy, origin, 1.0, 0.0, config=tight_cfg, return_debug=True
    )
    assert not tight_debug['step_obstacle_mask'][2, 2]
    assert not tight_debug['wall_obstacle_mask'][4, 4]

    wide_cfg = FusedESDFConfig(step_denoise_kernel=1, robot_z_bottom=-0.2, robot_z_top=4.0)
    _, _, _, wide_debug = build_fused_esdf_from_height(
        height, occupancy, origin, 1.0, 0.0, config=wide_cfg, return_debug=True
    )
    assert wide_debug['step_obstacle_mask'][2, 2]
    assert wide_debug['wall_obstacle_mask'][4, 4]

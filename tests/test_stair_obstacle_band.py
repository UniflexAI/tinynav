"""Raise the bottom only on pitched ascent; walls and platform checks remain active."""
from types import SimpleNamespace
import dataclasses

import numpy as np
import pytest
from std_msgs.msg import String

from tinynav.core.planning_node import PlanningNode, STAIR, build_obstacle_map
from tinynav.core.robot_specs import ObstacleConfig


def planner():
    normal = ObstacleConfig(dilation_cells=1)
    return SimpleNamespace(_stair_ascending=False, obstacle_config=normal,
        stair_obstacle_config=normal,
        stair_up_obstacle_config=dataclasses.replace(normal, robot_z_bottom=-.3))


def test_pitch_hysteresis_and_platform_restore_original_bottom():
    node = planner()
    for status, raised in [('up ok pitch=+14', True), ('up ok pitch=+10', True),
                           ('up search pitch=+7', False), ('up ok pitch=+10', False),
                           ('up ok pitch=+20', True), ('down ok pitch=-20', False)]:
        PlanningNode._on_stair_status(node, String(data=status))
        assert node._stair_ascending == raised
        assert PlanningNode.active_obstacle_config(node, True).robot_z_bottom == (-.3 if raised else -.4)
    node._stair_ascending = True
    assert PlanningNode.active_obstacle_config(node, False).robot_z_bottom == -.4


def test_missing_or_invalid_pitch_does_not_raise_bottom():
    for status in ['up ok', 'up ok pitch=bad', 'up ok pitch=nan', 'stopped']:
        node = planner()
        PlanningNode._on_stair_status(node, String(data=status))
        assert not node._stair_ascending


def test_raised_band_retains_vertical_wall_and_filters_lower_tread_span():
    grid = np.zeros((12, 12, 30))
    origin = np.array([-.3, -.3, -.75])
    zs = origin[2] + (np.arange(30) + .5) * .05
    grid[3, 3, (zs >= -.4) & (zs <= .025)] = .2
    grid[8, 8, (zs >= -.6) & (zs <= .6)] = .2
    base = ObstacleConfig(dilation_cells=0)
    raised = dataclasses.replace(base, robot_z_bottom=-.3)
    original = build_obstacle_map(grid, origin, .05, 0., base)
    ascent = build_obstacle_map(grid, origin, .05, 0., raised)
    assert original[3, 3] and not ascent[3, 3]
    assert original[8, 8] and ascent[8, 8]
    assert raised.robot_z_top == base.robot_z_top
    assert raised.min_wall_span_m == base.min_wall_span_m


def test_stair_wall_span_is_explicitly_point_three_and_tall_walls_remain():
    assert STAIR.min_wall_span_m==.3
    grid=np.zeros((4,4,30));origin=np.array([0.,0.,-.75])
    grid[1,1,10:16]=.2  # 0.25 m occupied span: ordinary riser remains traversable
    grid[2,2,8:17]=.2   # 0.40 m span: wall remains an obstacle
    cfg=ObstacleConfig(min_wall_span_m=STAIR.min_wall_span_m,dilation_cells=0)
    mask=build_obstacle_map(grid,origin,.05,0.,cfg)
    assert not mask[1,1] and mask[2,2]

"""Fixed camera-relative Go2 obstacle band in every stair phase."""
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
        stair_up_obstacle_config=dataclasses.replace(normal, robot_z_bottom=normal.robot_z_bottom + STAIR.up_bottom_raise_m))


def test_pitch_hysteresis_does_not_change_requested_band():
    node = planner()
    for status, raised in [('up ok pitch=+14', True), ('up ok pitch=+10', True),
                           ('up search pitch=+7', False), ('up ok pitch=+10', False),
                           ('up ok pitch=+20', True), ('down ok pitch=-20', False)]:
        PlanningNode._on_stair_status(node, String(data=status))
        assert node._stair_ascending == raised
        assert PlanningNode.active_obstacle_config(node, True).robot_z_bottom == -.2
        assert PlanningNode.active_obstacle_config(node, True).robot_z_top == .2
    node._stair_ascending = True
    assert PlanningNode.active_obstacle_config(node, False).robot_z_bottom == -.2


def test_missing_or_invalid_pitch_does_not_raise_bottom():
    for status in ['up ok', 'up ok pitch=bad', 'up ok pitch=nan', 'stopped']:
        node = planner()
        PlanningNode._on_stair_status(node, String(data=status))
        assert not node._stair_ascending


def test_requested_band_excludes_outside_points_and_keeps_full_height_wall():
    grid=np.zeros((12,12,30));origin=np.array([-.3,-.3,-.75])
    zs=origin[2]+(np.arange(30)+.5)*.05
    grid[3,3,(zs < -.2) | (zs > .2)]=.2
    grid[8,8,(zs >= -.6) & (zs <= .6)]=.2
    cfg=ObstacleConfig(dilation_cells=0)
    mask=build_obstacle_map(grid,origin,.05,0.,cfg)
    assert not mask[3,3] and mask[8,8]
    assert cfg.robot_z_bottom==-.2 and cfg.robot_z_top==.2
    assert cfg.min_wall_span_m==.3


def test_band_is_camera_relative_and_go2_modes_share_limits():
    from tinynav.core.robot_specs import GO2_CONFIG, GO2W_CONFIG
    assert STAIR.up_bottom_raise_m==0.
    for robot in (GO2_CONFIG,GO2W_CONFIG):
        assert robot.obstacle.robot_z_bottom==-.2 and robot.obstacle.robot_z_top==.2
    grid=np.zeros((4,4,30));origin=np.array([0.,0.,4.25])
    grid[1,1,:]=.2
    mask=build_obstacle_map(grid,origin,.05,5.,ObstacleConfig(dilation_cells=0))
    assert mask[1,1]


def test_stair_wall_span_is_explicitly_point_three_and_tall_walls_remain():
    assert STAIR.min_wall_span_m==.3
    grid=np.zeros((4,4,30));origin=np.array([0.,0.,-.75])
    grid[1,1,10:16]=.2  # 0.25 m occupied span: ordinary riser remains traversable
    grid[2,2,8:23]=.2   # full wall crosses the requested height window
    cfg=ObstacleConfig(min_wall_span_m=STAIR.min_wall_span_m,dilation_cells=0)
    mask=build_obstacle_map(grid,origin,.05,0.,cfg)
    assert not mask[1,1] and mask[2,2]


@pytest.mark.parametrize('stair', [True,False])
def test_unobserved_confirmed_wall_is_retained_only_in_stair_mode(monkeypatch,stair):
    from tinynav.core import planning_node
    fresh=np.zeros((4,4,4));grid=np.zeros_like(fresh);grid[1,1,1]=.2
    node=SimpleNamespace(origin=np.full(3,-.1),grid_shape=grid.shape,resolution=.05,step=1,
        occupancy_grid=grid,in_stair_mode=stair,obstacle_config=ObstacleConfig())
    monkeypatch.setattr(planning_node,'run_raycasting_loopy',lambda *a:fresh.copy())
    for _ in range(100):PlanningNode.update_occupancy_grid(node,None,np.eye(4),1.,1.,0.,0.)
    assert (node.occupancy_grid[1,1,1]>.1)==stair
    # An actual free-space observation must still remove a retained wall.
    fresh[1,1,1]=-.05
    for _ in range(5):PlanningNode.update_occupancy_grid(node,None,np.eye(4),1.,1.,0.,0.)
    assert node.occupancy_grid[1,1,1]<=0.

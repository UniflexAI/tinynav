# planner_rviz_sim

轻量级 Planner 仿真器（仅用于调试 `planning_node`）。

## 功能

该脚本会模拟一个机器人，并给 planner 提供最小必需输入：

- 发布 `/slam/odometry`（机器人位姿）
- 发布 `/slam/depth`（合成深度图，默认常量 2.5m）
- 发布 `/camera/camera/infra2/camera_info`
- 订阅 `/cmd_vel`（planner 输出速度）
- 订阅 RViz 的 `/goal_pose`（2D Goal Pose）并持续发布到 `/control/target_pose`
- 订阅 RViz 的 `/initialpose`（2D Pose Estimate）用于重置初始位姿

## 启动

> 需要 ROS2 humble + 本仓库 Python 依赖。

### Terminal A: 启动仿真器

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/xiaolefang/workspace/tinynav:/home/xiaolefang/workspace/tinynav/.venv/lib/python3.10/site-packages:$PYTHONPATH
python3 /home/xiaolefang/workspace/tinynav/tool/planner_rviz_sim.py
```

### Terminal B: 启动 planner

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/xiaolefang/workspace/tinynav:/home/xiaolefang/workspace/tinynav/.venv/lib/python3.10/site-packages:$PYTHONPATH
python3 -m tinynav.core.planning_node --sensor_source realsense
```

### Terminal C: 启动 RViz

```bash
source /opt/ros/humble/setup.bash
rviz2
```

建议配置：
- Fixed Frame = `world`
- Add 显示：
  - `Path` -> `/planning/trajectory_path`
  - `Odometry` -> `/slam/odometry`
  - `PointCloud2` -> `/planning/occupied_voxels`
  - `PointCloud2` -> `/planning/occupied_voxels_with_esdf`

交互：
- 用 `2D Goal Pose` 点击目标（会发布 `/goal_pose`）
- 仿真器会持续发布 `/control/target_pose`（避免 planner 只收到一次目标后无更新）
- 可用 `2D Pose Estimate` 重置机器人位置

## 说明

- 这是 planner 级仿真，不包含 perception/map 复杂链路。
- 深度图是简化的常量深度，用于验证控制逻辑和路径输出连通性。
- 若要更真实，可后续在脚本里叠加虚拟障碍深度模板。

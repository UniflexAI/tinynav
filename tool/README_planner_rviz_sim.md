# planner_rviz_sim

轻量级 Planner 仿真器（仅用于调试 `planning_node`）。

## 功能

该脚本会模拟一个机器人，并给 planner 提供最小必需输入：

- 发布 `/slam/odometry`（机器人位姿）
- 发布 `/slam/depth`（合成深度图，支持 flat/corridor/obstacles/stairs 场景）
- 发布 `/camera/camera/infra2/camera_info`
- 订阅 `/cmd_vel`（planner 输出速度）
- 订阅 RViz 的 `/goal_pose`（2D Goal Pose）并发布到 `/control/target_pose`
- 订阅 RViz 的 `/initialpose`（2D Pose Estimate）用于重置初始位姿

并支持两种目标发布模式：
- `--target_mode once`（默认）：点击一次只发一次 target（更接近真实 map_node 行为）
- `--target_mode continuous`：持续发 target（便于连通性排障）

## 启动

> 需要 ROS2 humble + 本仓库 Python 依赖。

### Terminal A: 启动仿真器

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/home/xiaolefang/workspace/tinynav:/home/xiaolefang/workspace/tinynav/.venv/lib/python3.10/site-packages:$PYTHONPATH
python3 /home/xiaolefang/workspace/tinynav/tool/planner_rviz_sim.py --scene stairs --target_mode once
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
- 默认只发一次 `/control/target_pose`（`--target_mode once`）
- 若调试链路，可切到 `--target_mode continuous`
- 可用 `2D Pose Estimate` 重置机器人位置

## 说明

- 这是 planner 级仿真，不包含 perception/map 复杂链路。
- 深度图是简化的常量深度，用于验证控制逻辑和路径输出连通性。
- 若要更真实，可后续在脚本里叠加虚拟障碍深度模板。

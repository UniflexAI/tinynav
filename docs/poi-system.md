# POI (Point of Interest) 系统文档

> 基于 main 分支 (`3c46210`) 整理

---

## 1. 概述

POI 是 tinynav 导航的目标点体系。整个流程围绕 `pois.json` 文件展开：**创建 POI → 下发导航目标 → 全局路径规划 → 局部避障 → 到达判定**。

核心数据结构非常简单——一个带三维坐标的命名点：

```json
{
  "0": {"id": 0, "name": "home",    "position": [0.0, 0.0, 0.0]},
  "1": {"id": 1, "name": "reception","position": [3.5, -1.2, 0.0]}
}
```

---

## 2. 数据格式

### pois.json

存储在地图目录下（如 `/tinynav/tinynav_db/maps/map_2025_01_01_12_00_00/pois.json`）。

```json
{
  "<id_str>": {
    "id": int,
    "name": str,
    "position": [x, y, z]   // 地图坐标系下的三维坐标，单位：米
  }
}
```

- `id_str` = `id` 的字符串形式，作为 JSON key
- `position` = `[x, y, z]`，地图坐标系（map frame）
- 建图完成后，系统自动在原点创建 `home` POI

---

## 3. POI 创建方式

### 3.1 viser 3D 编辑器（`tool/poi_editor.py`）

交互式 3D 可视化工具，基于 viser 库：

```bash
python tool/poi_editor.py --tinynav_map_path /path/to/map
```

功能：
- 在 3D 场景中添加/删除 POI 球体
- 通过 transform gizmo 拖拽调整位置
- 调整球体颜色和大小
- 保存到 `pois.json`

### 3.2 REST API（`app/backend/routers/poi.py`）

后端 HTTP 接口：

| Method | Path | 说明 |
|--------|------|------|
| `GET` | `/map/pois` | 列出所有 POI |
| `POST` | `/map/pois` | 创建 POI `{"name": "...", "position": [x,y,z]}` |
| `DELETE` | `/poi/{id}` | 删除指定 ID 的 POI |

### 3.3 建图后自动创建

建图完成时，`node_manager` 检查是否存在 `pois.json`，不存在则自动在原点创建 `home` POI。

---

## 4. 导航推演流程

### 4.1 整体架构

```
┌─────────────┐     /mapping/cmd_pois      ┌────────────┐
│  下发入口    │ ─────────────────────────→  │  map_node  │
│  (API/CLI)  │    JSON: {"0": {poi_dict}} │            │
└─────────────┘                             │ 重定位      │
                                            │ 坐标变换    │
                                            │ SDF 路径规划│
                                            │ 到达判定    │
                                            └─────┬──────┘
                                                  │
                    /control/target_pose            │  /mapping/global_plan
                    /mapping/poi                    │  /mapping/poi_change
                    /mapping/current_pose_in_map    │
                                                  ↓
                                            ┌────────────┐
                                            │planning_node│
                                            │ 局部避障    │
                                            │ 轨迹评分    │
                                            └─────┬──────┘
                                                  │
                    /planning/trajectory_path       │
                                                  ↓
                                            ┌────────────┐
                                            │  底层控制   │
                                            │ (cmd_vel)  │
                                            └────────────┘
```

### 4.2 详细推演步骤

#### Step 1: 下发导航目标

三种方式触发：

**A. REST API**
```
POST /nav/go-to-poi  {"poi_id": 1}
```
→ `node_manager.cmd_nav_start(poi_id=1)`
→ `_publish_cmd_pois(poi_id)` 读取 pois.json，将目标 POI 以 `{"0": poi_dict}` 格式发布到 `/mapping/cmd_pois`

**B. CLI 工具**
```bash
ros2 run tinynav pub_pois --tinynav_map_path /path/to/map --pois 2,1,0
```
→ 按指定顺序发布多个 POI（逗号分隔 ID），或省略 `--pois` 发布全部

**C. 直接发布 ROS 消息**
```python
# 向 /mapping/cmd_pois 发布 JSON String
payload = json.dumps({"0": pois["1"]})
```

#### Step 2: map_node 接收并解析

`map_node.pois_callback()` 接收 `/mapping/cmd_pois` 消息：

1. 解析 JSON → 提取所有 POI 的 position
2. 按 key 排序，重新索引为 `{0: position_0, 1: position_1, ...}`
3. 设置 `poi_index = 0`（从第一个开始）

#### Step 3: 重定位（Relocalization）

在每次关键帧回调 `keyframe_callback` 中：

1. **特征提取**：SuperPoint 提取当前帧特征点
2. **全局检索**：DINOv2 embedding 与预建地图 embedding 匹配（相似度阈值 0.85）
3. **局部匹配**：LightGlue 匹配当前帧与检索到的地图帧
4. **PnP 求解**：RANSAC + solvePnP 得到当前相机在地图坐标系下的位姿
5. **坐标变换求解**：计算 `T_from_map_to_odom`（map→odom 变换），通过位姿图优化（最近 100 个约束）

#### Step 4: 全局路径规划（SDF-based A*）

`try_publish_nav_path()` 在每次关键帧回调时执行：

1. **检查前置条件**：
   - `T_from_map_to_odom` 已计算（重定位成功）
   - `poi_index >= 0`（有活跃目标）
   - `poi_index < len(pois)`（未全部到达）

2. **计算当前位姿 in map frame**：
   ```
   pose_in_map = inv(T_from_map_to_odom) @ pose_in_odom
   ```

3. **到达判定**：
   - XY 平面距离 < 0.5m **且** Z 距离 < 2.0m → 判定到达
   - 到达后：`poi_index += 1`，发布 `/mapping/poi_change`，继续下一个 POI
   - 所有 POI 到达后：停止导航

4. **SDF 路径搜索**（`generate_nav_path_in_map`）：
   - 将起终点坐标转换为 SDF 体素索引
   - `search_close_to_sdf_map`：从起点/终点向 SDF < 0.2m 的自由空间搜索（避开障碍物附近的点）
   - `search_within_sdf_map`：A* 搜索，启发函数 = 欧几里得距离 + 20×Z 差异，约束 SDF < 0.2m
   - 拼接三段路径：起→自由空间 → 全局路径 → 自由空间→终点
   - 体素坐标转换回世界坐标

5. **目标点截取**：
   - 沿路径按 5 秒前瞻距离（max_speed × 5s = 2.5m）截取中间目标点
   - 转换到 odom 坐标系，发布到 `/control/target_pose`

6. **发布全局路径**：`/mapping/global_plan`（Path 消息，用于可视化）

#### Step 5: 局部避障与轨迹规划（planning_node）

`planning_node` 独立运行，以 ~10Hz 接收深度图 + 里程计：

1. **3D 射线投射建局部地图**：
   - 深度图 → 射线投射到 3D occupancy grid（100×100×10, 0.1m 分辨率）
   - 地图跟随机器人中心滚动更新

2. **2D 障碍物提取**：
   - 在机器人 Z 范围内（-0.2m ~ 0.5m）提取障碍物
   - 要求障碍物 Z 跨度 ≥ 0.4m（过滤地面凸起/楼梯）
   - 膨胀 3 个像素（安全余量）

3. **ESDF 计算**：`distance_transform_edt` → 2D ESDF map

4. **轨迹库生成**（`generate_trajectory_library_3d`）：
   - 采样加速度 × 角速度参数组合
   - 每条轨迹 2 秒时长，0.1s 步长
   - 输出：轨迹点序列 [x, y, z, qx, qy, qz, qw]

5. **轨迹评分**（`score_trajectories_by_ESDF`）：
   - 检查机器人足迹（中心 + 4 角）的 ESDF 最小值
   - 碰撞 → 无穷大分数；低于安全半径 → 惩罚

6. **最优轨迹选择**：
   - 代价函数 = 碰撞分 × 100000 + 目标距离 × 100 + 参数平滑性 × 10
   - 目标点来自 `/control/target_pose`（map_node 下发的中间目标）
   - POI 切换时（收到 `/mapping/poi_change`），暂停 3 秒等新路径

7. **发布**：最优轨迹 → `/planning/trajectory_path` → 底层控制

#### Step 6: 到达与切换

回到 map_node 的到达判定（Step 4.3）：

- 到达当前 POI → `poi_index += 1` → 发布 `/mapping/poi_change`
- planning_node 收到 poi_change → 暂停 3 秒 → 等待新目标
- map_node 下一个关键帧回调自动切换到下一个 POI
- 全部 POI 到达 → `poi_index = len(pois)` → 停止导航

---

## 5. 关键 ROS Topic 一览

| Topic | 方向 | 类型 | 说明 |
|-------|------|------|------|
| `/mapping/cmd_pois` | 外→map_node | `std_msgs/String` | 下发 POI 目标，JSON 格式 |
| `/mapping/poi` | map_node→外 | `Odometry` | 当前目标 POI 位姿（可视化用） |
| `/mapping/poi_change` | map_node→planning_node | `Odometry` | POI 切换信号 |
| `/mapping/current_pose_in_map` | map_node→外 | `Odometry` | 当前位姿 in map frame |
| `/mapping/global_plan` | map_node→外 | `Path` | 全局规划路径 |
| `/control/target_pose` | map_node→planning_node | `Odometry` | 中间目标点（odom frame） |
| `/planning/trajectory_path` | planning_node→控制 | `Path` | 局部避障轨迹 |
| `/map/relocalization` | map_node→外 | `Odometry` | 重定位结果（可视化） |

---

## 6. 工具与脚本

### pub_pois.py — 命令行发布 POI

```bash
# 发布所有 POI
python tool/pub_pois.py --tinynav_map_path /path/to/map

# 按指定顺序发布
python tool/pub_pois.py --tinynav_map_path /path/to/map --pois 2,1,0
```

发布到 `/mapping/cmd_pois`，等待 map_node 订阅上线后发送，5 秒超时。

### poi_editor.py — 3D 可视化编辑器

```bash
python tool/poi_editor.py --tinynav_map_path /path/to/map
```

功能：3D 场景中添加/拖拽/删除/保存 POI，同时可视化 SDF 地图、高斯泼溅/点云、重定位位姿、全局/局部规划路径。

---

## 7. 文件关系图

```
tinynav_map_path/
├── pois.json                 ← POI 定义（核心数据）
├── poses.npy                 ← 建图关键帧位姿
├── intrinsics.npy            ← 相机内参
├── occupancy_grid.npy        ← 3D 占据栅格 (0=未知, 1=自由, 2=占据)
├── occupancy_meta.npy        ← [origin_x, origin_y, origin_z, resolution]
├── sdf_map.npy               ← 3D SDF 距离场
├── splat.ply / pointcloud.ply← 3D 场景表示
└── ...
```

```
代码结构：
app/backend/routers/poi.py       ← REST API (CRUD)
app/backend/routers/nav.py       ← 导航触发 API
app/backend/node_manager.py      ← 桥接 API → ROS (cmd_pois 发布)
tool/pub_pois.py                 ← CLI 发布工具
tool/poi_editor.py               ← 3D 编辑器 (viser)
tinynav/core/map_node.py         ← 核心：重定位 + 全局规划 + 到达判定
tinynav/core/planning_node.py    ← 核心：局部避障 + 轨迹评分
```

---

## 8. 典型使用场景

### 场景 A: Web UI 发起导航

```
1. 用户在 Web UI 上点击 "前往 reception"
2. 前端调用 POST /nav/go-to-poi {"poi_id": 1}
3. node_manager 读取 pois.json，取出 POI 1
4. 发布 {"0": {"id":1, "name":"reception", "position":[3.5,-1.2,0.0]}} → /mapping/cmd_pois
5. map_node 收到 → 设为目标，开始规划
6. robot 导航至 reception
7. 到达判定通过 → 自动停泊
```

### 场景 B: 巡航多点

```bash
# 定义巡逻顺序：先去 2 号点，再去 1 号点，最后回 0 号点
python tool/pub_pois.py --tinynav_map_path /path/to/map --pois 2,1,0
```

map_node 会按 2→1→0 的顺序依次导航，每个点到达后自动切换到下一个。

### 场景 C: 取消导航

```
POST /nav/cancel
→ node_manager 发布 {} → /mapping/cmd_pois（清空目标）
→ map_node poi_index 重置，停止规划
```

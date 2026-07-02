# EKF 多源里程计融合方案

> 目标：在 `tool/` 下新增 `ekf_odom_node.py`，将多源里程计融合后输出 `/slam/odometry`，替代现有 `imu_propagator_node.py`。

---

## 1. 现状与问题

### 1.1 当前 `/slam/odometry` 数据流

```
/camera/camera/vio_20hz  ──→  perception_node.py  ──→  /slam/odometry_visual (20Hz)
                                                              │
/camera/camera/imu ────────→  imu_propagator_node.py ──→  /slam/odometry (100Hz)
```

或者通过 looper_bridge：

```
/camera/camera/vio_100hz  ──→  looper_bridge_node.py  ──→  /slam/odometry (100Hz)
```

**下游消费者**（均订阅 `/slam/odometry`，格式 `frame_id="world"`, `child_frame_id="camera"`）：

| 节点 | 用途 |
|------|------|
| `build_map_node.py` | 连续里程计记录，构建地图 |
| `map_node.py` | 导航定位 |
| `cmd_vel_control.py` | 底盘速度控制 |
| `poi_editor.py` | POI 编辑 |
| `qr_target_node.py` | 二维码引导导航 |

### 1.2 现有单源方案的问题

- 视觉 SLAM 单一依赖，暗光/纹理稀少场景精度下降，有累积漂移
- IMU 传播仅做短期插值，不纠正 SLAM 的绝对误差
- 轮式里程计、激光雷达里程计、QR 绝对定位、RTK 均未融合

---

## 2. 核心设计原则：职责分离

**`ekf_odom_node.py` 只做融合，不做坐标系转换。**

所有输入话题统一格式：
```
frame_id="world",  child_frame_id="camera"   （T_world_camera）
```

**坐标系转换由各数据源自己的 bridge 节点负责**，在发布到 EKF 输入话题前完成。

好处：
- EKF 节点无需任何传感器外参，逻辑纯粹
- 各 bridge 节点可独立调试（`ros2 topic echo` 直接对比各源输出）
- 新增传感器只需新增一个 bridge 节点，EKF 无需修改

---

## 3. 系统架构

```
原始数据                        Bridge 节点                         EKF 融合
                                （各自负责坐标系转换）
                                                               ┌──────────────────────┐
wheel encoder ──────────→  wheel_odom_bridge.py           ──→ │                      │
                            encoder Δpose → T_world_camera     │                      │
                                                               │   ekf_odom_node.py   │──→ /slam/odometry
/lidar_odom ────────────→  lidar_odom_bridge.py           ──→ │                      │     T_world_camera
                            lidar frame → T_world_camera       │   所有输入统一为      │
                                                               │   frame_id="world"   │
/slam/odometry_visual  ─────────────────────────────────────→ │   child="camera"     │
                            已是 T_world_camera，直连           │                      │
                                                               │                      │
AprilTag 检测 ──────────→  qr_odom.py                    ──→ │                      │
                            T_world_qrworld @ inv(T_cam_board) │                      │
                                                               │                      │
/fix (RTK GPS) ─────────→  rtk_odom.py                   ──→ │                      │
                            WGS-84 → ENU → T_world_camera      └──────────────────────┘
```

---

## 4. 各数据源分析

| 数据源 | EKF 输入话题 | 频率 | 在 EKF 中的角色 | 优点 | 缺点 |
|--------|-------------|------|----------------|------|------|
| 视觉 SLAM | `/slam/odometry` | 20 Hz | **Predict 步**（过程模型） | 高频连续，帧间相对变换稳定 | 长期漂移、暗光/重复纹理差 |
| 轮式里程计 | `/wheel/odom_camera` | 50–100 Hz | Update | 高频稳定，无视觉依赖 | 打滑/长期漂移 |
| 激光雷达里程计 | `/lidar/odom_camera` | 10–20 Hz | Update | 绝对定位准，光照无关 | 动态障碍物干扰 |
| QR 码定位 | `/qr/odom` | 间歇 1–5 Hz | Update（高置信） | mm 级精度，全局绝对 | 需标签可见 |
| RTK GPS | `/rtk/odom_camera` | 5–10 Hz | Update（仅位置） | 全局无漂移 | 无朝向，室内无信号 |

---

## 5. EKF 状态与算法（Error-State EKF，四元数姿态）

采用 **Error-State EKF（间接法）**：EKF 估计误差量 δx，标称状态 x̂ 单独传播，更新后将误差注入标称状态再重置。旋转用单位四元数表示，彻底消除欧拉角的万向节锁问题。

### 5.1 状态定义

**标称状态**（Nominal State，10 维）：

$$\hat{\mathbf{x}} = [\mathbf{p},\; \mathbf{v},\; \mathbf{q}]^\top \quad \mathbf{p},\mathbf{v} \in \mathbb{R}^3,\; \mathbf{q} \in \mathbb{H}_{unit}$$

- $\mathbf{p}$：相机原点在 world 帧中的位置
- $\mathbf{v}$：world 帧中的线速度
- $\mathbf{q}$：单位四元数 $[x,y,z,w]$（scipy 约定），表示 world→camera 旋转

**误差状态**（Error State，9 维，EKF 实际估计量）：

$$\delta\mathbf{x} = [\delta\mathbf{p},\; \delta\mathbf{v},\; \delta\boldsymbol{\theta}]^\top \quad \in \mathbb{R}^9$$

- $\delta\mathbf{p}$：位置误差
- $\delta\mathbf{v}$：速度误差
- $\delta\boldsymbol{\theta}$：旋转向量形式的姿态误差（$\mathbf{q}_{true} = \hat{\mathbf{q}} \otimes \text{Exp}(\delta\boldsymbol{\theta})$）

**协方差矩阵** $\mathbf{P} \in \mathbb{R}^{9 \times 9}$，定义在误差状态空间。

### 5.2 Predict 步（SLAM 里程计驱动）

SLAM 里程计 `/slam/odometry` 提供相邻帧间的相对变换 $T_{\Delta}$（4×4 SE3）。

**标称状态传播**（无噪声）：

$$\mathbf{p}' = \mathbf{p} + R(\hat{\mathbf{q}})\,\Delta\mathbf{p}$$
$$\mathbf{v}' = R(\hat{\mathbf{q}})\,\Delta\mathbf{p}\;/\;\Delta t$$
$$\mathbf{q}' = \hat{\mathbf{q}} \otimes \Delta\mathbf{q} \quad \text{（归一化）}$$

**误差状态 Jacobian** $\mathbf{F} \in \mathbb{R}^{9 \times 9}$（解析推导）：

$$\mathbf{F} = \begin{bmatrix}
\mathbf{I} & \mathbf{0} & -R\,[\Delta\mathbf{p}]_\times \\
\mathbf{0} & \mathbf{0} & -R\,[\Delta\mathbf{p}]_\times / \Delta t \\
\mathbf{0} & \mathbf{0} & \Delta R^\top
\end{bmatrix}$$

其中 $[\cdot]_\times$ 为反对称矩阵（skew）。

**协方差传播**：

$$\mathbf{P}' = \mathbf{F}\,\mathbf{P}\,\mathbf{F}^\top + \mathbf{Q}$$

### 5.3 Update 步（所有传感器通用）

测量值为 SE3 位姿（4×4），在误差状态空间构建创新向量：

```
δp_innov = p_meas - p_nom                         # 位置差
δθ_innov = (q_nom⁻¹ ⊗ q_meas).as_rotvec()        # 旋转向量形式（短路径）
innov    = [δp_innov, δθ_innov]                   # 6维
```

RTK 仅位置，创新为 `innov = p_meas - p_nom`（3 维）。

**观测矩阵**（误差状态空间，H 天然稀疏）：

$$\mathbf{H}_{6} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & \mathbf{0} & \mathbf{I}_3 \end{bmatrix} \in \mathbb{R}^{6 \times 9}, \qquad \mathbf{H}_{3} = \begin{bmatrix} \mathbf{I}_3 & \mathbf{0} & \mathbf{0} \end{bmatrix} \in \mathbb{R}^{3 \times 9}$$

**Mahalanobis 门控**：

$$d^2 = \boldsymbol{\nu}^\top \mathbf{S}^{-1} \boldsymbol{\nu}, \quad \mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^\top + \mathbf{R}$$

超过阈值则丢弃。

**Joseph form 更新**（数值稳定，保正定）：

$$\mathbf{K} = \mathbf{P}\mathbf{H}^\top\mathbf{S}^{-1}$$
$$\mathbf{P}' = (\mathbf{I} - \mathbf{K}\mathbf{H})\,\mathbf{P}\,(\mathbf{I} - \mathbf{K}\mathbf{H})^\top + \mathbf{K}\mathbf{R}\mathbf{K}^\top$$

**误差注入 + 重置**：

$$\mathbf{p} \mathrel{+}= \delta\hat{\mathbf{x}}_{0:3}, \quad
\mathbf{v} \mathrel{+}= \delta\hat{\mathbf{x}}_{3:6}, \quad
\mathbf{q} \leftarrow \mathbf{q} \otimes \text{Exp}(\delta\hat{\mathbf{x}}_{6:9}) \;\text{（归一化）}$$

### 5.4 噪声参数

**过程噪声 Q（误差状态顺序：δp, δv, δθ）：**

```
δp:  [0.01, 0.01, 0.005]  m²
δv:  [0.10, 0.10, 0.050]  (m/s)²
δθ:  [0.005, 0.005, 0.010] rad²
```

**测量噪声 R（位置 + 旋转向量对角线）：**

| 传感器 | 位置 m² | 旋转 rad² |
|--------|--------|---------|
| 轮式里程计 | 0.03, 0.03, 0.01 | 0.005, 0.005, 0.03 |
| 激光里程计 | 0.02, 0.02, 0.01 | 0.005, 0.005, 0.02 |
| QR 码      | 0.005, 0.005, 0.005 | 0.002, 0.002, 0.005 |
| RTK GPS   | 0.01, 0.01, 0.04 | （不观测） |

**Mahalanobis gate 阈值（χ² 分布，6/3 自由度）：**

| 传感器 | 阈值 | 参考置信度 |
|--------|------|----------|
| 轮式里程计 | 12.0 | ~97% |
| 激光里程计 | 16.0 | ~99% |
| QR 码 | 10.0 | ~95% |
| RTK GPS | 16.0 | ~99% |

---

## 6. Bridge 节点设计

### 6.1 QR 节点拆分：`qr_odom.py` + `qr_nav_node.py`

原 `qr_target_node.py` 同时承担检测、里程计发布、导航控制三项职责，且直接订阅 `/slam/odometry`。
拆分为两个独立节点，彻底消除循环依赖：

```
tag_mappose.json (T_map_qrworld)
tag_target.json  (T_qrworld_robot)
        │
        ├─→  qr_odom.py                          qr_nav_node.py
        │    每帧检测 → solvePnPRansac            订阅 /slam/odometry_fused
        │    T_camera_board (PnP 输出)            T_world_camera
        │    T_qrworld_camera = inv(T_cam_board)       ↓
        │    T_map_camera = T_map_qrworld @        T_map_world × T_world_camera
        │                   T_qrworld_camera            = T_map_camera
        │    T_world_camera = T_world_map @        T_map_robot = T_map_camera
        │                     T_map_camera                     @ T_CAMERA_ROBOT
        │         ↑ TF: world→map (map_node)            ↓
        │                                         T_robot_goal = inv(T_map_robot)
        └→  /qr/odom  (EKF update 源)                          @ T_map_goal
                                                       ↓ P 控制器
                                                 /control/cmd_vel
```

---

#### 6.1.1 `tool/qr_odom.py`

**职责**：每次成功检测到 AprilTag 板，发布 T_world_camera 给 EKF。**不订阅任何里程计**，无循环依赖。

**完整帧链**：

```
T_camera_board     ← solvePnPRansac（每帧检测）
T_qrworld_camera   = inv(T_camera_board)         （board 即 qrworld 原点）
T_map_camera       = T_map_qrworld @ T_qrworld_camera   （T_map_qrworld 预定义）
T_world_camera     = T_world_map @ T_map_camera   （T_world_map 来自 TF world→map）
```

**发布条件**：可见 tag 数 ≥ 2，且 solvePnPRansac 重投影误差 < 3px。

**协方差**：按重投影误差动态填写，误差越小协方差越小。

**标定依赖**：需在建图后保存 `tinynav_db/qrcode/tag_mappose.json`，内含 `T_map_qrworld`（见 6.4 节标定方法）。

---

#### 6.1.2 `tool/qr_nav_node.py`

**职责**：订阅 `/slam/odometry_fused`，将机器人当前位姿转换到 map 系，使用 P 控制器驱动到预定目标点。

**目标点（固定，不受 SLAM 漂移影响）**：

```python
T_map_goal = T_map_qrworld @ T_qrworld_robot   # 初始化时计算一次，之后固定
```

**当前位姿（map 系）**：

```python
T_world_camera  ← /slam/odometry_fused
T_world_map     ← TF lookup world→map
T_map_robot     = inv(T_world_map) @ T_world_camera @ T_CAMERA_ROBOT
```

**控制误差（机器人系）**：

```python
T_robot_goal = inv(T_map_robot) @ T_map_goal
dx, dy       = T_robot_goal[:2, 3]
bearing      = arctan2(dy, dx)        # 转向误差
dtheta       = arctan2(T_robot_goal[1,0], T_robot_goal[0,0])  # 最终朝向误差
```

控制增益与原 `qr_target_node.py` 保持一致：`K_LINEAR=0.5, K_ANGULAR=1.0`。

---

### 6.2 `tool/rtk_odom.py`

**职责**：将 RTK GPS（`sensor_msgs/NavSatFix`）转换为 T_world_camera，仅提供位置约束。

**坐标转换流程**：

```
WGS-84 (lat, lon, alt)
  └─→ 以首帧为原点，转换为局部 ENU (east, north, up)
       └─→ × T_world_enu（ENU 与 world 帧的对齐，初始化时设定）
            └─→ world 帧中的 GPS 天线位置
                 └─→ × inv(T_camera_gps)（固定外参）
                      └─→ T_world_camera → 发布 /rtk/odom_camera
```

**重要限制**：
- RTK 单天线**无朝向**，旋转协方差置为 999，EKF 仅更新 (x, y, z)
- 双天线 RTK 可同时提供朝向，此时全 6 维有效
- 室内/遮挡下信号丢失，依赖超时机制自动停用

---

### 6.3 `tool/lidar_odom_bridge.py`

**职责**：将激光里程计（lidar frame）转换为相机系，发布给 EKF。

**坐标转换**：

```
输入：T_odom_lidar（激光 SLAM 输出，frame_id="odom", child="lidar"）

T_world_lidar  = T_world_odom_lidar @ T_odom_lidar
                （T_world_odom_lidar：激光 odom 帧与 SLAM world 帧的对齐，首帧标定）

T_world_camera = T_world_lidar @ inv(T_camera_lidar)
                （T_camera_lidar：固定外参，从 URDF/标定获取）
```

**透传协方差**：若激光 SLAM 提供了有效协方差，直接透传给 EKF 使用，无需手动配置 R_lidar。

---

### 6.4 `wheel_odom_bridge.py`（可选）

若底盘轮式里程计话题为标准 `nav_msgs/Odometry`（`frame_id="odom"`, `child_frame_id="base_link"`），需转换为相机系：

```
T_world_camera_new = T_world_camera_prev @ T_camera_body @ ΔT_body @ inv(T_camera_body)
```

其中 `T_camera_body` 为固定外参，`ΔT_body` 为相邻帧轮式里程计增量。

若底盘直接发布增量（diff drive 常见），则无需维护上一帧，直接转换增量即可。

---

## 7. 文件布局

```
tool/
├── ekf_odom_node.py          # EKF 融合主节点
├── qr_odom/                  # QR 相关节点（子包）
│   ├── __init__.py
│   ├── record_node.py        # 一次性标定：保存 tag_target.json + tag_mappose.json
│   ├── odom_node.py          # QR 码绝对定位 bridge → /qr/odom
│   ├── nav_node.py           # QR 目标导航控制（订阅 /slam/odometry_fused）
│   ├── target_node.py        # 原 qr_target_node（保留，含旧 record/nav 模式）
│   └── generate.py           # 生成 QR 打印文件 + tag_satellite.json
├── rtk_odom.py               # RTK GPS bridge
├── lidar_odom_bridge.py      # 激光里程计坐标系 bridge
└── wheel_odom_bridge.py      # 轮式里程计坐标系 bridge（若需要）
tinynav/config/
└── ekf_odom.yaml             # EKF 噪声/阈值参数（无外参）
tinynav_db/qrcode/
├── tag_satellite.json        # tag ids, size, spacing（已有）
├── tag_target.json           # T_qrworld_robot（已有，record 模式保存）
└── tag_mappose.json          # T_map_qrworld（新增，建图后标定保存）
```

外参（T_camera_body、T_camera_lidar、T_camera_gps）各自存放在对应 bridge 的 yaml 或代码中，与 EKF 完全解耦。

---

## 8. 集成方式

### 8.1 替换 imu_propagator_node

```bash
# run_navigation.sh 中将：
uv run python /tinynav/tinynav/core/imu_propagator_node.py

# 替换为（按需启动各 bridge）：
uv run python /tinynav/tool/ekf_odom_node.py
uv run python /tinynav/tool/qr_odom/odom_node.py
uv run python /tinynav/tool/rtk_odom.py              # 室外 RTK 场景
uv run python /tinynav/tool/lidar_odom_bridge.py      # 有激光时

# 导航到 QR 目标（替换原 qr_target_node.py nav 模式）：
uv run python /tinynav/tool/qr_odom/nav_node.py
```

下游所有节点无需修改。

### 8.2 初始化策略

优先级：QR > RTK > 激光 > 视觉 SLAM

第一条任意源消息到达时设定初始状态和 P0，之后正常运行。初始化前不发布 `/slam/odometry`。

### 8.3 降级策略

| 场景 | 处理 |
|------|------|
| 轮式里程计超时 | 停止 Predict，仅靠 Update 维持（协方差增长） |
| 所有 Update 源超时 | 仅靠轮式里程计 Predict，写 warning 日志 |
| 全部源超时 | 停止发布，等待任意源恢复 |

---

## 9. `tag_mappose.json` 标定方法

`tag_mappose.json` 保存 `T_map_qrworld`（QR 板在 map 系中的固定位姿），需在建图完成后执行一次：

**标定流程**：

1. 完成建图，`map_node` 已发布稳定的 `world→map` TF
2. 将机器人停在 QR 板正前方，使用 `qr_target_node.py` record 模式检测板子
3. record 模式已保存 `T_qrworld_robot`（即 `tag_target.json`）
4. 同时从 TF 读取 `T_world_map`，从当前 `/slam/odometry` 读取 `T_world_camera`，计算：

```python
T_map_camera  = inv(T_world_map) @ T_world_camera
T_camera_board = solvePnP(...)           # 同帧检测结果
T_qrworld_camera = inv(T_camera_board)
T_map_qrworld  = T_map_camera @ inv(T_qrworld_camera)
# 或等价：
T_map_qrworld  = T_map_camera @ T_camera_board
```

5. 保存到 `tinynav_db/qrcode/tag_mappose.json`：

```json
{
  "T_map_qrworld": [[...], [...], [...], [...]]
}
```

> **注**：此标定只需做一次（地图不变则 T_map_qrworld 不变）。后续每次导航直接加载文件，无需重新标定。

---

## 10. 实现步骤

```
Phase 0 — QR 节点拆分（已完成）
  ✓ qr_odom.py：AprilTag → /qr/odom（T_world_camera）
  ✓ qr_nav_node.py：/slam/odometry_fused → /control/cmd_vel（map 系固定目标）
  ✓ record_node.py：一次采集 20 帧 → 保存 tag_target.json + tag_mappose.json

Phase 1 — EKF 核心（已完成）
  ✓ Error-State EKF，四元数姿态（NominalState + 9维误差状态 + Joseph form）
  ✓ 解析 Jacobian F（skew matrix 推导，非数值有限差分）
  ✓ 单元测试：35 项全通过（quat 工具、round-trip、predict、update 收敛、gate、P 正定性）

Phase 2 — 接入视觉 SLAM（~0.5天）
  □ 订阅 /slam/odometry_visual 做 Update
  □ 与原 imu_propagator_node 输出对比

Phase 3 — 接入轮式里程计（~1天）
  □ 实现 wheel_odom_bridge，标定 T_camera_body
  □ 验证遮挡相机时轮式里程计单独维持定位

Phase 4 — 接入激光 + QR + RTK（~1天）
  □ lidar_odom_bridge，标定 T_camera_lidar
  □ 扩展 qr_target_node record 模式保存 T_world_qrworld，实现 qr_odom.py
  □ rtk_odom.py，标定 T_camera_gps

Phase 5 — 调参与验证（~1天）
  □ rosbag 回放对比轨迹
  □ 调整 Q/R/gate 参数
```

---

## 附录 A：四元数姿态创新计算

Error-State EKF 不再需要欧拉角 wraparound。旋转创新通过四元数相对旋转计算，天生处理 ±π 歧义：

```python
q_err  = q_nom_inv ⊗ q_meas        # 相对旋转
if q_err.w < 0: q_err = -q_err     # 强制短路径（scalar part ≥ 0）
δθ = q_err.as_rotvec()             # 旋转向量 ∈ ℝ³，模 ≤ π
```

当 `q_nom` 和 `q_meas` 指向同一物理方向时（欧拉角相差 2π），`q_err ≈ identity`，`δθ ≈ 0`，创新自动为零——无需任何 wraparound 处理。

---

*文档版本：2026-06-29*

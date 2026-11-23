# App 控制与 stairs mode：更新、使用和真机升级

更新日期：2026-11-23；分支：`xiaole/stair-floors`。

## 本次交付

- 原生 TinyNav App 的 Operate 页显示前向相机/深度、实时局部 BEV、路径和机身轮廓；支持 2D/3D 切换。
- 修复设备地址带端口时 REST、相机和手柄 WebSocket 的连接；摇杆保持输入时每 100 ms 发送，释放和离开页面时发零速度。
- Stairs 设置页保存参数后可直接发起上楼/下楼；Operate → Stairs 对话框也可发起任务。仿真支持 1–9 层，每层固定 2 段；相机标定值固定。真机仍允许设置实际相机高度、每层段数、相对层数和转弯方向。
- 修复最后平台有侧向偏移时被过窄候选带排除、持续 `no_seed` 的情况。只允许修正到可达、已观测且满足高度/净空条件的平台；没有支撑仍停车，最后计数后不会另起一次转弯。
- 保留带噪测试、暂停/继续、重置、第三方视角和独立安全断言。完整 SLAM 扫图建图试验不在本次交付中。

## 2026-11-23 真机反馈修复

控制转换已修复原地路径因浮点残差产生倒退/大角速度的问题；五个真机 bag 的 5,652 条路径回归通过。当前 Go2/Go2W 障碍高度带已统一为 bottom -0.2 m、top +0.2 m，上楼不再额外抬高下界（替代此前的上楼抬高 10 cm 策略）。弱转向置信度平台会以 `landing reason=turn_uncertain` 停车等待明确转向，不再继续搜索平地开口。带噪 UP 9 / DOWN 9 重新通过；新增开放门洞测试仍未通过，见[详细结果与范围](stair_field_control_fix.md)。本次是 Python 修改，重启后端和导航子进程即可加载，不需因本次修改单独重建 Flutter。

## 3090 仿真使用

| 入口 | 地址 | 用途 |
| --- | --- | --- |
| 原生 TinyNav App | <http://192.168.11.116:18083/> | Operate → Stairs → 9 层 → Up / Down；或首页 Stairs 设置页直接启动 |
| 仿真控制页 | <http://192.168.11.116:18080/> | 设置噪声、速度、层数、平台进深，重置环境 |
| 第三方视角 | <http://192.168.11.116:18082/> | 自由视角、跟随、全景；观察同一物理实例 |

Operate 顶部可选 color / depth，下面是实时 BEV。所有页面共享一个环境，发起新仿真任务会重置到该方向的起点。
摇杆会取消仿真自动任务，松手归零，超过 0.5 秒未收到手柄命令也归零。停止/断言失败后需重新启动任务或重置。
这个 0.5 秒超时由仿真驱动实现，不代表真机执行器新增了独立失联保护。

该环境使用 MuJoCo 深度与带噪仿真 odom，调用原有 stair/planning/control 方法；不是视觉 SLAM 定位验收，也不是分布式 ROS 链路验收。楼梯模式无需先扫完整地图或设置全局 POI。详细仿真部署见 [MuJoCo App 部署](stair_mujoco_app.md) 和 [原生 App 使用](stair_tinynav_app_view.md)。

## 上下九层验证

两次任务均从原生 App 发起，楼梯净宽/平台进深均为 1.2 m，10 个整层平台、18 段楼梯。
噪声为 `combined / scale 0.5 / seed 7`：每轴位置标准差 5 mm，yaw 标准差 0.25°，pitch 随机标准差 0.25° 加 1.7 Hz、峰值 0.5° 的周期扰动。见 [噪声模型](stair_odom_noise_trial.md)。

| 任务 | 计数 | 仿真任务耗时（含停稳） | 最小井口间距 | 结果 |
| --- | --- | --- | --- | --- |
| DOWN 9 | 依次 1–18 | 379.52 s | 0.05717 m | 通过 |
| UP 9 | 依次 1–18 | 370.02 s | 0.08903 m | 通过 |

完成还要求最终平台位置正确、速度归零、持续停稳 1.5 秒且漂移小于 0.12 m。跌倒、碰墙、机身越井口、计数/高度、终点和停车断言全部保留；离线仿真中 11 个断言的 AST 与 `efc25a2` 相同。
本次发布复核：六个相关 Python 套件 **77 passed**，Flutter **3 passed**；静态分析 **0 error、0 warning、41 条 info**，两个 Web 构建通过。具体命令、日志位置与本次结果见 [发布验证记录](validation/stair_app_release_2026_11_21.json)；完整任务事件见 [App 控制验证](validation/stair_app_controls_2026_11_21.json)。

## 真机是否只要切分支后 Flutter build？

**还需要更新并重启 Python 后端/导航进程。** 仅重新构建前端不能加载 `stair_target.py` 的修复。若现有真机依赖、ROS、相机/VIO 和执行器链路已经正常，通常不需要重装系统、重建所有 TensorRT 模型或重做全局地图。

1. 先结束当前运动任务，停止原有 App 后端及它管理的导航子进程，备份数据目录中的 `stair_settings.json`。在实际部署使用的源码目录更新；保留已有本地改动，不要强制覆盖：

   ```bash
   git status --short
   git fetch origin
   git switch xiaole/stair-floors
   git pull --ff-only origin xiaole/stair-floors
   git rev-parse HEAD
   ```

2. 在同一份源码、原有 Flutter 环境中构建普通 App：

   ```bash
   cd app/frontend
   flutter pub get
   flutter build web --release --no-web-resources-cdn
   ```

   真机不要传 `--dart-define=TINYNAV_SIMULATION=true`；不要沿用 3090 仿真 IP 的 `TINYNAV_DEFAULT_DEVICE`。在 App 填实际后端地址，支持 `IP:端口`，不写端口时默认为 8000。Android/iOS 则构建并安装对应平台应用，Web 构建不会更新已安装的手机应用。

3. 用原有 systemd / Docker / tmux 启动方式重启后端及导航进程，确认服务实际加载本次源码。真机服务环境应**不设置** `TINYNAV_APP_PROFILE=mujoco`，也不要用 `scripts/run_mujoco_app.sh`。可在启动后端的 shell 执行 `unset TINYNAV_APP_PROFILE`；若由服务管理器配置环境，需修改那里。源码在镜像内部时要重建/重新部署镜像；绑定挂载源码时通常重启即可。环境变量或镜像改变则需按原部署方式重建容器。

4. 刷新浏览器，必要时清理旧站点缓存；确认实际后端的 `/device/info` 没有 `simulation` 能力、`/sensor/mode` 是预期传感器，查看 `/nav/stair/settings`。不要把真机 App 连到 3090 的仿真后端。

5. 真实链路仍须提供 `/slam/depth`、连续的 `/slam/odometry_visual` / `/slam/odometry` 和匹配的深度 CameraInfo。Looper 使用模块已有 VIO，经本分支 bridge 接入；厂商驱动及 `/insight_full` 必须先可发现，避免被自动识别成 RealSense。ROS 域/网卡使用实际设备配置，不能照搬仿真的 localhost-only 域。

6. App → Stairs 填**实际相机离地高度**、每层段数和相对层数；不能把仿真的 0.4669 m 当作真机标定。普通 U 型楼梯一层通常为两段，`floors=1` 不是一段。Operate → Stairs 可按现场选择 Auto / Left / Right。确认实际 `/cmd_vel` 执行器和唯一命令来源；已有执行器桥接时不要再重复启动 App 管理的 Unitree 桥接。

MuJoCo 的 ONNX 步态和知道井口真值的独立断言不会部署到真机。真实运动控制、相机外参及停车链路仍按 [真机试验与录包说明](stair_floors_physical_trial.md) 检查；首次用有限的小任务验证，不直接以仿真九层成功推断真机九层可行。新机器完整安装见 [Ubuntu x64 部署](stair_floors_ubuntu_x64_deployment.md)。

## 真机录包回传

在原生 App 的 **Operate → 相机上方固定录包栏**，点 **Record bag** 后再发任务，红色 `REC` 显示计时。停止运动后保留一小段静止数据，点 **Stop & save**，等待 `Saved · N messages`。文件夹图标可查看并复制后端保存路径；`Check inputs` 表示缺少关键输入消息，请查看详情。

一键录包已包含原始传感器、派生 `/slam/*`、`/planning/height_map` 和控制/楼梯状态，无需另开终端补录。**必须同时更新后端并重新构建 Flutter**。文件保存在机器人/后端主机，不会自动下载到浏览器。回传完整 rosbag 目录和 `info.json`，另带实际相机高度/外参、控制模式、任务设置、楼梯尺寸及异常时间点；详见[真机录包文档](stair_floors_physical_trial.md#recording-needed-for-analysis)。

仿真连接下该栏显示自动保存轨迹的提示，现场 rosbag 按钮禁用；仿真 JSON/NPZ 与真机 rosbag 是两种记录。


## 2026-11-23：VIO 自动恢复与楼梯最低速度

更新 Python 后端、楼梯/规划/cmd_vel/Unitree 进程，并重新构建 Flutter。
VIO 跳变、重启或断流先停车清图，健康输入持续合格后自动重建并继续；不再
要求 App 确认。单独暂停的任务仍暂停。上下楼非零前进最低 0.25 m/s；
`min_wall_span_m` 显式为 0.3 m，上一版实际也为 0.3，并非本次从 0.2 提高。
完整规则和验证结果见 [VIO 自动恢复说明](stair_vio_safety.md)。


### 固定障碍高度带补充更新

Go2/Go2W 的普通导航、平台、上楼和下楼现在统一使用相对相机世界 Z 的
`bottom=-0.2 m, top=+0.2 m`。上楼额外抬高量归零，min_wall 仍为 0.3。
此次未修改 Flutter，更新 Python 并重启后端/规划进程即可加载这一项。

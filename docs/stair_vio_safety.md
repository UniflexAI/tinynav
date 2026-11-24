# VIO 自动恢复、楼梯最低速度与墙体阈值

更新：2026-11-23；分支：`xiaole/stair-floors`。本文替代 `7555516` 中的人工确认流程。
Go2/Go2W 最新障碍高度带固定为相对相机世界 Z 的 **[-0.2, +0.2] m**，上楼不再额外抬高下界。

旧版验证记录保留在 `validation/stair_vio_safety_2026_11_23.json`，不能用旧结果代表本版。

## VIO 恢复不再要求点击确认

米级位置跳变、VIO RESTARTING/NOT_INITED/LOST、odom/depth 断流仍会立即停止
命令，并清除楼梯点云历史、梯段世界坐标、平台目标、规划占据图、轨迹和旧 cmd。
定位、深度和已启用的 VIO 状态持续新鲜且正常至少 2 秒后，监管节点自动解除
故障等待，再次更换地图 epoch；只允许新坐标、新深度、新目标进入重建。
规划至少积累两帧新深度后才输出路径。当前任务继续，楼层计数保留并重新对齐
高度基准，坐标重置本身不增加楼层计数。

App Operate 只显示 **Automatic recovery · waiting for stable inputs**。
恢复后提示自动消失，没有确认按钮或恢复对话框。先前单独暂停或取消的任务
不会被 VIO 恢复自动启动；普通任务暂停仍由用户正常操作。

| 检测 | 处理 |
| --- | --- |
| 相邻原始位置差 ≥1 m | 停车、清图，持续健康后自动恢复 |
| VIO 重启或运行后离开正常跟踪状态 | 同上 |
| odom 接收间隔 >0.5 s、depth >0.75 s | 同上 |
| 曾出现的 VIO 状态心跳中断 >3 s | 同上 |
| 源时间倒退 >1 s、比 ROS 时钟陈旧 >1 s 或超前 >0.1 s | 同上，时钟和新数据合格后恢复 |
| 较小跳变：位移 >0.1 m 且速度 >3 m/s，或源时间间隔 >1 s | 清图，等待 2 s 且当前输入合格后恢复 |
| 监管节点心跳丢失 >1 s | 控制和 Unitree 桥拒绝运动；监管恢复并通过健康检查后继续 |

重复/乱序旧帧不刷新健康度；一张恢复的深度帧不足以解除硬故障等待。
新的中断或跳变会重新开始观察。监管状态仍持久保存在数据目录
`vio_safety.json`，监管进程重启后也走自动恢复检查，不能靠旧缓存直接运动。

`GET /nav/vio` 和 `/nav/vio_safety` 带 `recovery_mode=automatic`、地图 epoch、
故障编号和等待状态。旧确认接口保留兼容入口，但不能绕过健康检查或触发恢复；
新 App 不调用它。自动恢复发生在监管节点，不依赖前端在线。

## 上下楼最低前进命令为 0.25 m/s

楼梯模式的有效非零前进命令最低 0.25 m/s，覆盖 up/down、平台搜索、带线速度
的转弯和最终登台。零速度、原地转弯、故障、暂停、到达和路径超时仍可以输出零。
规划要求原地转弯或线速度归零时，控制立即给零，避免加速度平滑留下 0.19 m/s
等不可执行的前行残量。越过起步死区会直接给到最低可执行速度；高于下限的
变化仍沿用原加速度限制。普通导航的速度下限和固定倒退词表不变。

规划同步排除 0～0.25 m/s 之间的正速度候选，并增加恰为 0.25 的候选，避免
预测低速、执行时却强行放大。带前进的角速度候选限制在 0.2 rad/s 内；更急的
转向采用原地旋转再前进，避免提高线速度后在窄平台切入井口。
最终登台仍使用 0.25，下调到 1 秒滚动预测窗口，
让近目标可以由短距离候选到达；平台搜索使用 1.5 秒窗口，梯段仍为 3 秒。候选全部使用相同的足迹、
ESDF 和碰撞评分，未降低 0.4 m 登台距离、图有效期或井口/接触断言。
JIT 采样函数和签名保留，只在 Python 调用层组合和过滤候选。

没有增加 PID，没有修改真实 Unitree 步态。最低命令不保证真机实际速度，仍需
通过真机反馈和录像验证死区、执行延迟、打滑及刹停距离。

## min_wall 为 0.3 m：核对结果而非新提高

用户所指的 `min_wall` 在本仓库中名为 `min_wall_span_m`：占据格内的竖直高度
跨度达到该值才判为障碍。**上一版 `7555516` 已经是 0.3 m**，楼梯模式此前
继承的也是 0.3；本次只把楼梯配置显式固定到 0.3，并打印：

```text
Stair obstacles: min_wall_span=0.3m
```

因此不能把它描述成从 0.2 提高到 0.3，也不能声称由此解决了真机台阶误判。
安全半径中的 0.2、楼梯搜索高度图的 wall_span、相机高度和 z-bottom 都是不同
参数。当前 Go2/Go2W 的普通导航、平台、上楼和下楼均使用 bottom -0.2 m、
top +0.2 m，`up_bottom_raise_m=0`，避免上楼时再次抬成 -0.1 m。
min_wall 仍为 0.3 m。较窄的窗口只用于规划占据图的障碍判定；楼梯目标生成器的
实测高度图、机身足迹和独立几何断言保留。B2 等显式配置保持原值。
若真机仍误判，需核对实际运行 SHA、启动日志、深度/odom 对齐和障碍图。

## 部署和录包

停止任务后更新本分支，重启后端、楼梯、规划、cmd_vel 和 Unitree Python 进程，
同时重新构建 Flutter。只刷新页面或只 build 前端都不会加载后端恢复逻辑。
App 自动启动一个持久监管节点；同一 ROS 域只运行一个监管实例。
独立 tmux 楼梯脚本自行启动监管，不能与 App 启动器混用。

App 数据卷应保留 `vio_safety.json`。监管使用 ROS 时钟校验传感器源时间戳，
现场需校时。新 Operate 录包仍请求 38 个 topic，包含 `/nav/vio_safety`，
录包和分析步骤见 [真机试验说明](stair_floors_physical_trial.md)。

当前 3090 原版 App 为 `http://192.168.11.116:18083/`，仿真后端为 `:18081`。
部署后刷新页面；Operate 不再要求 Confirm VIO recovery。

## 验证范围

原始五个 bag 只读分析，不把数据播放进执行器。回归原始 odom/depth/VIO 状态
时序、自动恢复观察期、两个图的失效、旧目标清理和阻塞期间零命令。
图缓存使用明确标记测试清理，不等同于全部深度重规划或修后真机运动轨迹。
下楼录包包含人工干预且没有准确接管时间，不能把所有异常归因于自主控制。

仿真保留原有倾倒、墙/护栏接触、1.0 × 0.5 m 独立矩形井口边界、计数高度、
最终位置和停稳断言；另检查无死区命令、清图、自动恢复及故障期间零命令。
数据为 combined / scale 0.5 / seed 7，楼梯与平台均 1.2 m，10 个整层平台。
全程故障矩阵为 up/down × none/jump/dropout/restart，另在梯段中途注入故障。
测试工具不发送解除故障的人工确认。

```bash
python -m tool.field_vio_regression --bags /bags --out /tmp/five-bags-auto.json
python -u tool/stair_mode_sim.py --sim-root /simrepo \
  --direction up --floors 9 --levels 10 --flight-width 1.2 --landing-depth 1.2 \
  --odom-noise combined --noise-scale .5 --noise-seed 7 \
  --vio-fault dropout --fault-at 3 --max-sim-seconds 700 --result-json /tmp/up-auto.json
```

源坐标平移跳变和 VIO 重启状态受到覆盖；任意缓慢漂移、纯姿态跳变、DDS/SDK
故障及实际刹停距离不在本次闭环证明范围。已有开放门洞平台问题仍未宣称解决。


## 761cffb 验证结果（高度带调整前）

相关 Python 回归 189 passed；排除的两项旧后端测试
`test_render_metadata_origin`、`test_odom_yaw_90` 在旧基线下也失败，未修改其期望。
Flutter 10 passed，web build 成功，analyze 0 errors / 0 warnings（40 条 info）。
这不是全库测试全绿或真实 SDK/机器狗闭环成功的声明。

8 组完整任务均顺序计到 18 段、到达并通过停稳；6 组在 15 s 的梯段中途注入
jump/dropout/restart 的一层任务也全部完成两段并停稳。原 12 个几何/终点断言
全部保留，当前脚本共有 21 个断言；全轨迹检查没有落入 0～0.25 m/s 死区的
非零前进输出，带前进的角速度没有超出 0.2 rad/s。

| 任务 | 故障 | 含停稳仿真耗时 | 最小独立井口间距 | 结果 |
| --- | --- | ---: | ---: | --- |
| UP 9 | none | 449.60 s | 0.179050 m | 通过 |
| UP 9 | jump | 456.60 s | 0.135548 m | 通过 |
| UP 9 | dropout | 450.10 s | 0.284570 m | 通过 |
| UP 9 | restart | 455.10 s | 0.099207 m | 通过 |
| DOWN 9 | none | 462.60 s | 0.333551 m | 通过 |
| DOWN 9 | jump | 464.60 s | 0.303268 m | 通过 |
| DOWN 9 | dropout | 479.60 s | 0.258721 m | 通过 |
| DOWN 9 | restart | 473.10 s | 0.232733 m | 通过 |

相较旧版，更多采用原地转向，耗时有所增加。表中间距来自独立机身矩形检查，
不能作为真机安全余量。之前提速后切入井口的失败保留在 `r2/down-*.json`；
新回归没有删除失败证据、扩大边界或缩小检查机身。

五个 bag 的原始时序回归和 5,652 条 Path 控制转换回归均通过。VIO 恢复后可以
看到自动换 epoch 和清图；11:36 的末尾仍未恢复合格输入，因此继续保持停车，
没有凭超时强行启动。11:41 的输入恢复后自动继续，不再要求确认。
这些计数不能替代修后真机运动实测。

[机器可读记录](validation/stair_auto_recovery_2026_11_23.json) 包含参数、源码
SHA256、各场景结果和速度轨迹检查。服务器原始工件保存在
`/home/dm/tinynav-stair-dev/experiments/auto-recovery-20261123/`，本版导航快照为
`r4-source/`，结果/日志/NPZ 为 `r4/`；早期失败和被替代的测试也保留。


## 固定高度带更新

本轮在 `761cffb` 基础上调整为 Go2/Go2W 全阶段 [-0.2,+0.2] m。
需重启 Python 后端和规划进程；本次未改 Flutter。
自动 VIO 恢复、0.25 m/s 最低非零前进命令、平台转向约束和 min_wall=0.3 保持。
上述旧版 8 组故障矩阵记录不作为新高度带已复测的声明。


固定高度带的首轮 UP 9 在第七层平台出现后右腿与端墙接触，原断言中止运行。
诊断保存了接触几何、位姿和占据栅格：端墙转到相机背后后，已确认占据仍按每帧
0.99 衰减，接触前墙体值约 0.09，低于 0.1 判据，障碍掩码消失。
现仅在楼梯模式保留未被当前射线观测到的已确认占据；实际空闲射线仍能清除它，
滚动窗口和 VIO epoch 清图仍生效。普通导航的时间衰减保留。
这项修复保持所要求的 [-0.2,+0.2] 高度带和原几何断言，不靠扩大窗口通过测试。


修复后的固定源码 `source-r2` 下，相关 Python 测试 192 passed，另两项既有基线
失败仍明确排除；新增测试验证视野外占据保留、真实空闲观测清除及普通导航衰减。
本轮重新运行的加噪九层结果如下，原 12 个几何/终点断言全部保留：

| 任务 | 含停稳仿真耗时 | 最小独立井口间距 | 结果 |
| --- | ---: | ---: | --- |
| UP 9 | 447.60 s | 0.179050 m | 通过 |
| DOWN 9 | 479.60 s | 0.337926 m | 通过 |

本轮未重跑此前的八组故障矩阵或五个原始 bag，不复用其结果作为新高度带验证。
没有修后真机运动实测。机器可读结果见
[固定高度带验证记录](validation/stair_obstacle_band_020_2026_11_23.json)。
源码、初次失败、接触诊断及新结果保存在服务器
`/home/dm/tinynav-stair-dev/experiments/band-020-20261123/`，最终结果在 `r2/`。

## 2026-11-24 field follow-up

See [today's field audit and target correction](stair_field_2026_11_24.md). The target correction rejects measured railing/wall crossings without changing the VIO recovery policy, speed settings or obstacle band. All three Nov24 bags include manual intervention. Noisy UP9/DOWN9 and 169 focused tests pass; physical dead-zone calibration and false stair obstacles remain open. `min_wall=0.4` is not deployed because it empties the current narrow-band obstacle map.

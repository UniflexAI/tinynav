# VIO 故障停车、清图与 App 人工恢复

日期：2026-11-23；分支：`xiaole/stair-floors`。

## 行为与恢复

`vio_safety_node` 独立监控原始 `/slam/odometry`、`/slam/depth` 和
`/camera/camera/vio_status`，10 Hz 发布带持久故障编号的 `/nav/vio_safety`。
米级跳变或 VIO 重启会锁定停车，楼梯点云历史、梯段世界坐标、平台目标、规划
占据图、轨迹和旧 cmd 一起失效。App BEV 收到空图，避免旧图看起来仍可使用。
任务楼层计数保留；恢复后的第一份定位只用于重新对齐计数高度基准，不把坐标
重置当成上下楼进度。新目标只能从恢复后的新数据生成。

| 检测 | 处理 |
| --- | --- |
| 连续原始位置差 ≥1 m | 锁定停车、清图 |
| VIO RESTARTING、已运行后的非 TRACKING/TRACKING_STATIC | 锁定停车、清图 |
| odom 接收间隔 >0.5 s、depth >0.75 s | 锁定停车、清图 |
| 曾收到的 VIO 状态心跳中断 >3 s | 锁定停车、清图 |
| 源时间倒退 >1 s，或相对 ROS 时钟陈旧 >1 s / 超前 >0.1 s | 锁定停车、清图 |
| 较小跳变：位移 >0.1 m 且速度 >3 m/s，或源时间间隔 >1 s | 清图并等待稳定 2 s；已有锁定保持 |
| 监管节点心跳丢失 >1 s | App、轨迹控制和 Unitree 桥均拒绝运动 |

重复或乱序的旧时间戳不刷新输入健康度。坏深度和非有限位置不能用于解锁。
传感器与主机时钟需同步；本规则不检测所有缓慢漂移、纯姿态跳变或标定错误。

1. 发生故障后，Operate 顶部出现红色 **VIO safety stop** 栏，保持停车。
2. 等待定位、深度和已启用的 VIO 状态恢复，且最后异常后连续稳定至少 2 s。
   **Recovered · manual confirmation required** 表示可以确认，仍不会自动运动。
3. 核对机器狗姿态、所在平台/梯段和周围环境后，点击 **Confirm VIO recovery**，
   再点对话框的 **Confirm recovery**。**Keep stopped** 保持锁定。
4. 后端向监管节点提交当前 `fault_id`。准备不足、旧编号或监管节点失联都拒绝。
   成功后再换一个图版本，只接受确认时刻之后的定位、深度、目标和轨迹；规划
   至少积累两帧新深度后才输出路径。已有任务可继续；单独暂停过的任务仍暂停。

普通 Continue、手动摇杆和重发楼梯任务不能绕过锁定。锁定状态保存在 App 数据
目录 `vio_safety.json`；重启监管节点也要求人工确认。不要删文件绕过恢复流程。
同一数据目录有进程锁，避免重复监管；同一 ROS 域也只应运行一个权威监管节点。

## 部署与接口

需同时更新后端/楼梯/规划/cmd_vel/Unitree Python 进程并重建 Flutter；只更新
前端或只重启页面不够。App 后端自动启动监管节点，数据库目录与 App 一致。
独立 `scripts/run_looper_stair_navigation.sh` 自行启动监管，不能与 App 启动器混用。
手动拆分运行时，先启动以下节点，状态文件放在持久数据卷中：

```bash
uv run python /tinynav/tinynav/core/vio_safety_node.py \
  --state-file /tinynav_data/vio_safety.json
```

示例数据路径应替换为该设备实际 App 数据目录。`/nav/vio_safety` 使用可靠、
TRANSIENT_LOCAL、深度 1 的 QoS；重连消费者也会收到最近状态。
REST `GET /nav/vio` 查询，`POST /nav/vio/confirm` 的 JSON 为
`{"fault_id": 当前整数编号}`。真实后端调用 ROS
`/vio_safety/set_parameters_atomically` 的 `confirm_fault_id`，不是仅修改网页状态。
新 Operate 录包请求 38 个 topic，包含 `/nav/vio_safety`。旧五个 bag 不含此 topic，
回归工具依据原始传感器数据重新运行监管逻辑。

Unitree 桥在 Move 入口和 0.1 s 定时器中检查监管状态，锁定时调用 StopMove。
本次没有接真机 SDK；模拟客户端测试不能证明 SDK 延迟、丢包或物理刹停距离，
也没有加入 PID 或修改真实步态。实际停止仍须在现场验证。

## 回归方法与范围

五个 bag 从 `/home/dm/Desktop/debug` 只读读取，不运行 `ros2 bag play`，不创建
Unitree 执行器。按照记录到达顺序处理 odom/depth/VIO 状态，使用真实 ROS 监管
回调、两类图清理回调和 cmd_vel 门控，检查锁定不自解、清旧图和零命令。
图缓存用明确标记检查失效，**不是**把全部深度重新闭环规划，更不是修后真机轨迹。
有人工介入的下楼段仍保留，不能据此将每次定位异常归因于自主运动。

```bash
python -m tool.field_vio_regression --bags /bags --out /tmp/five-bags-vio.json
python -m tool.field_control_regression --bags /bags --out /tmp/five-bags-control.json
python -u tool/stair_mode_sim.py --sim-root /simrepo \
  --direction up --levels 10 --floors 9 --flight-width 1.2 --landing-depth 1.2 \
  --odom-noise combined --noise-scale .5 --noise-seed 7 \
  --vio-fault jump --fault-at 3 --max-sim-seconds 650 --result-json /tmp/up-jump.json
```

仿真矩阵为 `up/down × none/jump/dropout/restart`。jump 在估计位姿中永久平移
2 m；dropout 断 odom/depth 2 s；restart 同时注入 RESTARTING。真值位置不跳，
原有墙/护栏接触、井口矩形、倾倒、计数高度、最终位置和停稳断言保留。
故障测试另检查停车时限、锁定漂移、清图、提前确认和旧编号被拒、人工确认前
不能完成任务。测试脚本显式模拟人工确认；生产系统不会定时替人确认。

故障恢复回归也暴露了下楼末端台阶离开视野时，逐帧中心线随相机侧摆的问题。
现在上下楼都保留实测梯段线，避免把相机侧摆反复叠加到目标；仍只能选已观测
且可达的支撑。清图会丢弃保留线。

恢复延迟会改变进入台阶的相位，Auto 原先的可见面积均值在首平台可能未达 3。
没有降低此阈值或取消弱方向停车：新增独立的连续实测侧墙证据，要求机身已在
梯段、已有实测台阶轴、一侧至少 10 个高于梯段平面的实墙格且超过另一侧 3 倍，
同一方向持续 1.5 s；歧义、缺帧或反向会重新计时。新梯段重置该证据；清图丢弃未完成的证据区间，已确认的左/右拓扑与楼层计数
一起保留，它不是旧坐标目标。
梯段中途断流补测还暴露最终登台的小距离规划问题：粗速度采样最小正值为
0.167 m/s，三秒候选会越过近目标，规划宁可选择静止，随后实测平地记忆过期。
现在仅在 `finishing` 阶段增加接近机器人最小可执行速度的候选（Go2 采样 0.12 m/s，
给弧线路径的纵向投影留余量，控制下限仍为 0.1 m/s），
照常做同样的碰撞评分；控制层也只在这一阶段取消梯段 0.25 m/s 的提速，并给最小速度判定增加
1e-6 m/s 的数值容差，避免 0.1 的浮点投影残差被当成不可执行速度而归零。
没有降低 0.4 m 登台距离、净空、地图有效期或独立井口/接触断言。

证据仍不足时，原有 `turn_uncertain` 锁定停车继续有效。

最终结果见 [机器可读验证记录](validation/stair_vio_safety_2026_11_23.json)。
原始 bag 不变。服务器工件保存在
`/home/dm/tinynav-stair-dev/experiments/vio-lock-20261123/`，包括失败、诊断和各版
回归；最终选定源码快照与其测试单独记录，不能将较早通过结果冒充最终验证。
已有开放门洞越界/终点 no_seed 问题见
[上一轮控制修复说明](stair_field_control_fix.md)，本次不宣称解决这些场景。


## 单元、接口与五包结果

最终相关 Python 回归 177 passed；Flutter 10 passed、web build 成功，analyze
0 errors / 0 warnings（41 条 info）。ROS 实测接口回归覆盖保留消息、新订阅者、
标准参数服务和 App HTTP 409/422；Unitree 门控使用模拟 SportClient，未发真机命令。

较宽的测试选择中另有两项旧后端测试失败：`test_render_metadata_origin`、
`test_odom_yaw_90`，在原始 `e4daa36` 源码下也复现同样失败，最终相关回归明确
排除它们，没有修改期望或标记通过。全库还有两个旧测试导入已不存在的
`uf_find` / `theta_star` 导致收集失败，因此不能把相关回归表述为全库测试全绿。

| 真机 bag | 原始时序回归 |
| --- | --- |
| 11:34:58 | 两次较小跳变清图等待；无硬锁定 |
| 11:36:42 | 137.912 s、147.920 s 米级跳变锁定；随后超时与 VIO 重启，结束仍未就绪，拒绝确认 |
| 11:41:12 | 84.065 s 米级跳变锁定；输入恢复仍保持锁定，旧编号拒绝，测试末尾显式确认可解锁 |
| 11:43:36 | 无误触发硬锁定 |
| 11:46:11 | 无误触发硬锁定 |

五包均通过清图/旧目标失效/零命令/不自动解锁检查。11:41 的测试确认是工具显式
模拟的操作，不是从旧录包推断原机曾确认。控制转换的 5,652 条原始 Path 回归也
再次通过，原地误倒退归零、真实倒退路径保留；不代表真机轨迹已得到复测。


## 最终闭环结果

同一固定源码快照 `release-source-v16` 下，8 组完整任务全部顺序计到 18 段并到达、
通过停稳检查；三类故障的上下楼梯段中途补测 6/6 也全部完成一层（两段）。

| 方向 | 故障 | 含停稳仿真耗时 | 最小独立井口间距 | 结果 |
| --- | --- | ---: | ---: | --- |
| UP 9 | none | 366.60 s | 0.156429 m | 通过 |
| UP 9 | jump | 375.10 s | 0.160461 m | 通过 |
| UP 9 | dropout | 372.10 s | 0.180895 m | 通过 |
| UP 9 | restart | 373.10 s | 0.160461 m | 通过 |
| DOWN 9 | none | 375.60 s | 0.103685 m | 通过 |
| DOWN 9 | jump | 379.10 s | 0.081276 m | 通过 |
| DOWN 9 | dropout | 378.10 s | 0.085661 m | 通过 |
| DOWN 9 | restart | 380.60 s | 0.091101 m | 通过 |

这里的间距来自 1.0 × 0.5 m 独立机身矩形检查，不是真机安全裕度承诺。
原 12 个断言 AST 全部保留，总计 20 个断言。最终 JSON、日志、NPZ 在上述工件
根目录的 `release-v16/`；源码 SHA256 见验证记录和 `release-v16-manifest.json`。
当前 App 在 `http://192.168.11.116:18083/`，仿真后端为 `:18081`。

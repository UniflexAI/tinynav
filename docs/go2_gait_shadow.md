# Go2 gait read-only commissioning

The semantic navigation process still emits the existing velocity topic. In a
separate terminal with the existing Unitree SDK and ONNX Runtime CPU installed,
run the gait on the real IMU and motor feedback without sending motor traffic:

```bash
cd /tinynav
python3 -m tool.go2_gait_shadow --iface <robot-network-interface> \
  --sim-dir /path/to/tinynav-sim/mujoco --seconds 30
```

Use tinynav-sim commit 7d63a42cdf71b4f672cfac3cda1ad6b26c3cfe3a and its
go2_blind_v3.onnx. The log prints q, dq, and (with fresh velocity input) the
candidate position targets, all in SDK order. The node only subscribes to
rt/lowstate and rt/cmd_vel. Reading stationary data verifies reception and
plausibility. Verify physical joint directions against observed individual
joint movements and their encoder traces before enabling any motor writer.

Go2 SDK leg order is FR, FL, RR, RL. The ONNX order is FL, FR, RL, RR.
Each leg has hip, thigh and calf in that order. The simulation test checks
the named axes against the Unitree Go2 model and checks all 12 output targets
against the existing MuJoCo runtime. It simulates a zero velocity input upon
the injection of a command timeout, feedback timeout or emergency stop and
verifies that later packets cannot automatically clear the latch. This branch
provides shadow mode and does not validate physical motor stop behavior.

# Path tracking controller

`tinynav.platforms.cmd_vel_control` follows `/planning/trajectory_path` using
feedback from `/slam/odometry`. The controller samples the planned trajectory
at elapsed time, combines its forward/yaw feedforward with longitudinal and
heading PID corrections, and adds a lateral-position correction. Odometry is
converted from the camera pose to the same control center used by the planner,
using `ROBOT_CONFIG.cam_offset_3d`.

Strictly increasing pose timestamps define the trajectory spacing. The current
planner gives every pose the same timestamp, so those paths use the existing
`planner_dt * path_pose_stride` fallback (1 second). Yaw interpolation takes the
short route across +/-pi. Sampling stops at the last pose; it does not extrapolate.

## Tuning

Start the existing node with optional ROS parameters:

```bash
python -m tinynav.platforms.cmd_vel_control --ros-args \
  -p tracking_linear_pid:="[1.2, 0.1, 0.05]" \
  -p tracking_angular_pid:="[1.8, 0.05, 0.05]" \
  -p tracking_lateral_gain:=1.5
```

PID arrays are `[Kp, Ki, Kd]`; all gains must be finite and non-negative.
The integral is clamped to +/-0.5 and the derivative uses a 0.1-second low-pass
filter. Increase proportional gains for tighter correction and reduce them if
commands oscillate. Integral terms remove persistent offsets; derivative terms
damp changes. Tune against the selected robot's response and configured speed
limits.

Pause and navigation deactivation retain their existing behavior. Inactive
navigation yields `/cmd_vel` to teleoperation. Missing/stale odometry (0.5 seconds)
and invalid poses stop output. The existing adaptive stale-path slowdown/stop,
forward acceleration limit, and robot minimum/maximum speeds still apply.
Explicit planner reverse segments retain the fixed straight reverse command.
PID state resets on pause, deactivation, reverse, or a stale/invalid input.

## CPU validation

The controller tests need ROS 2 Humble, NumPy, and SciPy; no perception models or
GPU are required for these tests. In an Ubuntu 22.04 ROS 2 environment:

```bash
sudo apt-get install ros-humble-ros-base python3-pip
python3 -m pip install numpy==1.26.1 scipy==1.15.3
source /opt/ros/humble/setup.bash
PYTHONPATH=. ROS_LOCALHOST_ONLY=1 python3 -m unittest discover \
  -s tests -p 'test_*tracking.py' -v
```

Tests cover straight and curved kinematic recovery, coordinate transformations,
angle wrapping, path timestamps, bounded PID state, robot limits, camera offset,
pause/deactivation, stale/invalid inputs, fixed reverse, and ROS message delivery.
The ROS node closed-loop test includes the actual acceleration/minimum-speed
logic. Kinematic tests exercise controller behavior without modeling robot dynamics.

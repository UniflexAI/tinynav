# Go2 rig (quadruped)

The Unitree Go2 walking rig for the tinynav gz-sim simulator
(`scripts/run_simulator.sh --robot go2`). Worlds are robot-free
(tool/simulator/worlds/: empty.sdf / depot.sdf / factory.sdf); every robot
spawns from tool/simulator/robots/<type>/ -- the launcher keeps the per-world
spawn-pose table (a teleport after VIO init poisons the ISAM graph, so the
pose must be right at create time). The lekiwi cylinder stays the
`--robot lekiwi` default.

## What lives here

Everything is plain files: no colcon, no compiled packages, nothing to
build.

- `go2_tinynav.xacro` — the whole robot in one file: chassis, legs and the
  D435i sensor head on the dog's nose. Visuals are the collision
  primitives (near-black, meshless). The head keeps the lekiwi contract
  verbatim (stereo 544x480 @15 Hz, baseline 0.051 m, RGB, 200 Hz IMU with
  the camera-convention rotation, topics `/camera/camera/...`), so
  perception_node and camera_info_publisher cannot tell the robots apart.
- `go2_controller.py` — the trot controller, single process. Boots straight
  into a treadmill trot; `/cmd_vel` only adds velocity on top. Carries two
  safety patches: a cmd_vel clamp (0.045/0.015 m/s — the open-loop plant
  has ~13x gain, the PI servo closes the loop, the clamp is only a
  ceiling) and sqrt/D clamps in the analytic IK (over-reach used to raise
  math domain errors and leave gz holding stale joint commands).
- `ros_control.yaml` — controller_manager config for
  GazeboSimROS2ControlPlugin (joint_state_broadcaster +
  joint_group_position_controller); spawn.sh passes its absolute path into
  the xacro as `ctl_config:=`.
- `spawn.sh` — spawn chain in the order that works:
  robot_state_publisher **before** `create` (GazeboSimROS2ControlPlugin
  reads robot_description from its parameter API), spawners after the
  create, the trot controller last.
- `gt_twist_pub.py` / `cmd_vel_servo.py` — the velocity loop:
  gz ground truth finite-differenced into a body-frame twist, PI servo
  (20 Hz, params `kp_x/ki_x/kp_w/ki_w`) feeding the plant's `/robot1/cmd_vel`.
  Desired velocity goes in on `/cmd_vel` (absolute); direct publishers on
  `/robot1/cmd_vel` bypass the servo. Measured tracking: vx/wz commands are
  followed within ±0.014 from the first second.

## Gait caveat (read before trusting speeds)

The trot gait is a skating gait: stance feet sweep at 15.3x the commanded
speed and the robot glides on friction. It is stable and, behind the servo,
accurate — but it is not a calibrated platform, and reverse tracks at ~73%
(forward gain ~13x, backward ~9x). Do not "fix" the gait math against textbook kinematics:
the effective gains come from the stance-foot skating arithmetic, so a naive
rescale double-compensates and the robot topples (seen in earlier tuning). In-place turns
also translate the robot (anisotropic friction), a few meters per long
turn is normal.

## Run

`run_simulator.sh --robot go2` brings the whole stack up (the launcher also
runs tool/simulator/kill_sim.sh first, so a previous rig never leaks into
the next one). Manual:

```bash
source /opt/ros/humble/setup.bash
TINYNAV_WORLD_NAME=empty bash /tinynav/tool/simulator/robots/go2/spawn.sh
```

Spawn height matters: a 0.8 m drop flips the robot (soft position servo +
touchdown transient); keep it under ~0.4 m above the standing surface.

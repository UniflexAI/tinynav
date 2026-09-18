#!/usr/bin/env bash
# Bring the Go2 up inside an already-running gz server (tinynav sim rig).
#
# Order constraints:
#   robot_state_publisher MUST be up before `create` -- the
#   GazeboSimROS2ControlPlugin reads robot_description from its parameter
#   API; spawners only after the create, the trot controller last.
#
# Driven inputs (all env, set by scripts/run_simulator.sh --robot go2):
#   TINYNAV_WORLD_NAME  gz world name (create -world)
#   TINYNAV_SPAWN_X/Y/Z/YAW  spawn pose from the launcher's per-world table
#                   (0.8 m drop flips the robot: keep drop height sane)
#   TINYNAV_ROBOT_DIR  this directory
# no set -u before the ROS setup: it references unset vars
# (AMENT_TRACE_SETUP_FILES)

DIR=${TINYNAV_ROBOT_DIR:-/tinynav/tool/simulator/robots/go2}
WORLD_NAME=${TINYNAV_WORLD_NAME:?TINYNAV_WORLD_NAME not set}
SPAWN_Z=${TINYNAV_SPAWN_Z:-0.35}
NS=/robot1

command -v ros2 >/dev/null 2>&1 || source /opt/ros/humble/setup.bash
set -u

cd "$DIR"
echo "[go2] xacro"
# ctl_config: the merged rig is a single self-contained xacro; the
# GazeboSimROS2ControlPlugin still needs the controller yaml by absolute path
ros2 run xacro xacro "$DIR/go2_tinynav.xacro" robot_name:=robot1 \
  ctl_config:=$DIR/ros_control.yaml > /tmp/go2_robot.urdf \
  || { echo "[go2] XACRO FAIL"; exit 1; }

echo "[go2] robot_state_publisher (must precede create)"
ros2 run robot_state_publisher robot_state_publisher /tmp/go2_robot.urdf \
  --ros-args -r __ns:=$NS > /tmp/go2_rsp.log 2>&1 &
RSP=$!
sleep 3

echo "[go2] spawn (z=$SPAWN_Z, world=$WORLD_NAME)"
# spawn from the FILE, not from $NS/robot_description: the -topic path
# mis-places gz camera sensors (they end up inside the trunk mesh; every
# sensor pose in the URDF is ignored).
# ros2_control still gets the model from robot_state_publisher (ROS side).
timeout 60 ros2 run ros_gz_sim create -world "$WORLD_NAME" \
  -file /tmp/go2_robot.urdf -name go2 -z "$SPAWN_Z" \
  || { echo "[go2] SPAWN FAIL"; kill $RSP; exit 1; }
sleep 8

echo "[go2] controllers"
timeout 45 ros2 run controller_manager spawner joint_state_broadcaster --ros-args -r __ns:=$NS \
  || echo "[go2] JSB FAIL"
timeout 45 ros2 run controller_manager spawner joint_group_controller --ros-args -r __ns:=$NS \
  || echo "[go2] JGC FAIL"

echo "[go2] trot controller (single process, no colcon) + truth twist + PI servo"
python3 "$DIR/go2_controller.py" --ros-args -r __ns:=$NS \
  > /tmp/go2_cpg.log 2>&1 &
python3 "$DIR/gt_twist_pub.py" --ros-args -r __ns:=$NS \
  > /tmp/go2_gt.log 2>&1 &
python3 "$DIR/cmd_vel_servo.py" --ros-args -r __ns:=$NS \
  > /tmp/go2_servo.log 2>&1 &
sleep 5

echo "[go2] up -- desired velocity goes to /cmd_vel (absolute; the servo output is $NS/cmd_vel)"
wait $RSP

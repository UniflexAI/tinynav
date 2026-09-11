#!/bin/bash
# TinyNav gz-sim launcher: one named tmux window per component.
#
# Usage:  bash /tinynav/scripts/run_simulator.sh --map
#
# Default: sim + perception + planning + teleop; pass --map to also start
#          map_node and the rviz goal relay (localization + arrow goals).
#
# Attach:  tmux attach -t tinynav_sim

SESSION=tinynav_sim
WITH_MAP=0
for arg in "$@"; do
  case $arg in
    --map) WITH_MAP=1 ;;
    *) echo "usage: bash $0 [--map]"; exit 1 ;;
  esac
done
cd /tinynav
mkdir -p logs

command -v ros2 >/dev/null 2>&1 || source /opt/ros/humble/setup.bash
[[ -f /3rdparty/message_filters_ws/install/local_setup.bash ]] && source /3rdparty/message_filters_ws/install/local_setup.bash

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  export __GLX_VENDOR_LIBRARY_NAME=nvidia
  export __NV_PRIME_RENDER_OFFLOAD=1
  [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
else
  echo "WARN: no NVIDIA GPU detected; sensor rendering may fail on old Mesa"
fi
export GDK_SCALE=1

msg_bridge_args="\
/camera/camera/infra1/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/infra2/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/color/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU \
/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist"

MAP_DIR=/tinynav/output/map_gaz_color
NAV_DB_DIR=/tinynav/output/nav_sim
mkdir -p "$NAV_DB_DIR"

win() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -d -t "$SESSION" -n "$1" -c /tinynav "$2"
  else
    tmux new-session -d -s "$SESSION" -n "$1" -c /tinynav "$2"
  fi
  tmux set-option -w -t "$SESSION:$1" remain-on-exit on
  tmux set-option -w -t "$SESSION:$1" automatic-rename off
}

win gz "ign gazebo -s -r --headless-rendering -v 4 tool/simulator/robot_scene.sdf 2>&1 | tee logs/gz.log"
sleep 1
win gui "ign gazebo -g -v 3 2>&1 | tee logs/gzgui.log"
win bridge "ros2 run ros_gz_bridge parameter_bridge $msg_bridge_args 2>&1 | tee logs/bridge.log"
win caminfo "uv run python tool/simulator/camera_info_publisher.py 2>&1 | tee logs/caminfo.log"
win percept "uv run python tinynav/core/perception_node.py 2>&1 | tee logs/perception.log"
win planning "uv run python tinynav/core/planning_node.py 2>&1 | tee logs/planning.log"
win control "uv run python tinynav/platforms/simulator_control.py 2>&1 | tee logs/control.log"
win teleop "uv run python tinynav/platforms/keyboard_teleop.py 2>&1 | tee logs/teleop.log"
win rviz "rviz2 -d /tinynav/docs/vis.rviz 2>&1 | tee logs/rviz.log"
if [[ $WITH_MAP == 1 ]]; then
  win map "uv run python tinynav/core/map_node.py --tinynav_map_path $MAP_DIR --tinynav_db_path $NAV_DB_DIR 2>&1 | tee logs/map.log"
  win relay "uv run python tool/rviz_goal_to_poi.py --map-dir $MAP_DIR 2>&1 | tee logs/relay.log"
fi

echo "session '$SESSION' up (WITH_MAP=$WITH_MAP):"
tmux list-windows -t "$SESSION" -F '  #{window_index}:#{window_name}'
echo "attach: docker exec -t tinynav tmux attach -t $SESSION   (Ctrl+B D detach)"

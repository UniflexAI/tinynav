#!/bin/bash
# TinyNav gz-sim launcher: one named tmux window per component.
#
# Usage:  bash /tinynav/scripts/run_simulator.sh [--map] [--world <sdf>] [--auto <scene>]
#
# Default: sim (empty world) + perception + planning + teleop; --map also starts
#          map_node and the rviz goal relay (localization + arrow goals).
#          --world: base world SDF (default tool/simulator/worlds/robot_scene_empty.sdf;
#                   use tool/simulator/worlds/robot_scene.sdf for the depot factory).
#          --auto:  scripted scene (tool/simulator/gazebo_scene, e.g. l_corridor):
#                   spawns obstacles, resets robot to origin, publishes targets.
#
# Ground-truth relocalization: without --map, sim_gt_reloc corrects
# /slam/odometry_visual into the gazebo world frame using the chassis ground
# truth from /pose/info, so scene runs start exactly at the origin without
# respawning perception. With --map, map_node's own relocalization is used
# and sim_gt_reloc is off.
#
# Attach:  tmux attach -t tinynav_sim

SESSION=tinynav_sim
WITH_MAP=0
WORLD_SDF=tool/simulator/worlds/robot_scene_empty.sdf
AUTO_SCENE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --map) WITH_MAP=1; shift ;;
    --world) WORLD_SDF="$2"; shift 2 ;;
    --auto) AUTO_SCENE="$2"; shift 2 ;;
    *) echo "usage: bash $0 [--map] [--world <sdf>] [--auto <scene>]"; exit 1 ;;
  esac
done
cd /tinynav
mkdir -p logs

WORLD_NAME=$(grep -oP '(?<=<world name=")[^"]+' "$WORLD_SDF" | head -1)
[[ -z $WORLD_NAME ]] && { echo "no <world name> in $WORLD_SDF"; exit 1; }

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
/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
/world/$WORLD_NAME/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"

# Procedural textures must exist before gz loads the world SDF
uv run python tool/simulator/gazebo_scene/gen_textures.py

MAP_DIR=/tinynav/output/map_gaz_color
NAV_DB_DIR=/tinynav/output/nav_sim
mkdir -p "$NAV_DB_DIR"

win() {
  # Start an interactive shell so the pane stays usable after Ctrl+C or process exit.
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -d -t "$SESSION" -n "$1" -c /tinynav "exec bash -i"
  else
    tmux new-session -d -s "$SESSION" -n "$1" -c /tinynav "exec bash -i"
  fi
  tmux set-option -w -t "$SESSION:$1" remain-on-exit on
  tmux set-option -w -t "$SESSION:$1" automatic-rename off
  tmux send-keys -t "$SESSION:$1" "$2" Enter
}

win gz "ign gazebo -s -r --headless-rendering -v 4 $WORLD_SDF 2>&1 | tee logs/gz.log"
sleep 1
win gui "ign gazebo -g -v 3 2>&1 | tee logs/gzgui.log"
win bridge "ros2 run ros_gz_bridge parameter_bridge $msg_bridge_args 2>&1 | tee logs/bridge.log"
win caminfo "uv run python tool/simulator/gazebo_scene/camera_info_publisher.py 2>&1 | tee logs/caminfo.log"
PERCEPT_ARGS=""
if [[ $WITH_MAP == 0 ]]; then
  # sim_gt_reloc republishes the SLAM odometry corrected into the gazebo
  # world frame; perception emits its raw stream on ..._odometry_visual_raw.
  PERCEPT_ARGS="--ros-args -r /slam/odometry_visual:=/slam/odometry_visual_raw"
  win reloc "uv run python tool/simulator/gazebo_scene/sim_gt_reloc.py 2>&1 | tee logs/reloc.log"
fi
win percept "uv run python tinynav/core/perception_node.py $PERCEPT_ARGS 2>&1 | tee logs/perception.log"
win planning "uv run python tinynav/core/planning_node.py 2>&1 | tee logs/planning.log"
win control "uv run python tinynav/platforms/simulator_control.py 2>&1 | tee logs/control.log"
win teleop "uv run python tinynav/platforms/keyboard_teleop.py 2>&1 | tee logs/teleop.log"
win rviz "rviz2 -d /tinynav/docs/vis.rviz 2>&1 | tee logs/rviz.log"
if [[ $WITH_MAP == 1 ]]; then
  win map "uv run python tinynav/core/map_node.py --tinynav_map_path $MAP_DIR --tinynav_db_path $NAV_DB_DIR 2>&1 | tee logs/map.log"
fi
if [[ -n $AUTO_SCENE ]]; then
  AUTO_ARGS=""
  [[ $WITH_MAP == 1 ]] && AUTO_ARGS="--no-reloc"  # map_node is the authority
  win gz_scene "uv run python tool/simulator/gazebo_scene/scene_runner.py $AUTO_SCENE $AUTO_ARGS 2>&1 | tee logs/gz_scene.log"
fi

echo "session '$SESSION' up (WITH_MAP=$WITH_MAP, WORLD=$WORLD_SDF, AUTO=${AUTO_SCENE:-none}):"
tmux list-windows -t "$SESSION" -F '  #{window_index}:#{window_name}'
echo "attach: docker exec -t tinynav tmux attach -t $SESSION   (Ctrl+B D detach)"

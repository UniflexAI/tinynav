#!/bin/bash
# TinyNav gz-sim launcher: one named tmux window per component.
#
# Usage:  bash /tinynav/scripts/run_simulator.sh [--stack full|sensor] [--map] [--world <sdf>] [--auto <scene>] [--db <path>] [--robot lekiwi|go2]
#          --db: exported as TINYNAV_DB_PATH to every window, so nodes that log
#          via tinynav.core.logsetup land in the same data root as pilot's.
#
# Default (--stack full): sim (depot factory) + perception + planning + teleop;
#          --map also starts map_node and the rviz goal relay (localization +
#          arrow goals).
#          --world: base world SDF (default tool/simulator/worlds/empty.sdf).
#                   Worlds carry NO robot: empty.sdf / depot.sdf / factory.sdf /
#                   yard.sdf are pure environments, the robot spawns by type at
#                   run time. yard.sdf = empty base + static textured anchor
#                   walls: the go2's high camera needs far-field anchors or the
#                   VIO sinks while driving on empty's repetitive floor -- pick
#                   it for free driving; scripted scenes bring their own walls.
#          --auto:  scripted scene (tool/simulator/gazebo_scene, e.g. l_corridor):
#                   spawns obstacles, resets robot to origin, publishes targets.
#
# --robot go2: quadruped rig instead of the lekiwi cylinder. The Go2 carries
#          the same D435i sensor head (identical topics/rates/baseline) but is
#          a walking robot: it spawns from URDF (tool/simulator/robots/go2/,
#          see that directory's README) and desired velocity goes to
#          /cmd_vel through the PI velocity servo (plant input: /robot1/cmd_vel).
#          Spawn poses live in the per-world table below (go2 spawns crouched
#          at 0.26: the ros2_control initial_value holds the trot stance from
#          spawn, so the pre-controller window is bent-leg and ground-clear).
#
# --stack sensor: only the sensor face (gz/gui/bridge/caminfo/percept/control),
#          for when pilot owns the rest — pilot spawns planning itself, and its
#          map_node is the localization authority, so neither planning nor
#          sim_gt_reloc may run here (double planner / double reloc). Conflicts
#          with --map and --auto for the same reason.
#
# Ground-truth goal transformation: in the full stack without --map,
# sim_gt_reloc substitutes for map_node's localization authority: it
# broadcasts TF world->map (gz world standing in for the built map) and
# /map/relocalization (exact, from ground truth), and continuously transforms
# scene goals from gz world coordinates into perception's raw SLAM frame
# (scene_runner publishes them gz-authored on /sim/target_pose_gz). planning
# stays entirely in the raw SLAM frame: its odometry input is remapped to the
# raw stream below. With --map, map_node's own relocalization is used and
# sim_gt_reloc is off. (In the sensor stack the raw stream is left alone --
# pilot's map_node consumes it.)
#
# Attach:  tmux attach -t tinynav_sim

SESSION=tinynav_sim
WITH_MAP=0
# defaults: the go2 rig on the featureless empty world; yard (textured anchor
# walls) is the drive-test pick via --world.
ROBOT=go2
WORLD_SDF=""
WORLD_SET=0
AUTO_SCENE=""
STACK=full
while [[ $# -gt 0 ]]; do
  case $1 in
    --stack) STACK="$2"; shift 2 ;;
    --map) WITH_MAP=1; shift ;;
    --world) WORLD_SDF="$2"; WORLD_SET=1; shift 2 ;;
    --auto) AUTO_SCENE="$2"; shift 2 ;;
    --db) DB_PATH="$2"; shift 2 ;;
    --robot) ROBOT="$2"; shift 2 ;;
    *) echo "usage: bash $0 [--stack full|sensor] [--map] [--world <sdf>] [--auto <scene>] [--db <path>] [--robot lekiwi|go2]"; exit 1 ;;
  esac
done
[[ $ROBOT != lekiwi && $ROBOT != go2 ]] && { echo "--robot must be 'lekiwi' or 'go2'"; exit 1; }
# default world: the featureless empty baseline; yard adds textured anchors
# for go2 drive tests (far-field structure keeps the VIO from sinking)
if [[ $WORLD_SET == 0 ]]; then
  WORLD_SDF=tool/simulator/worlds/empty.sdf
fi
[[ $STACK != full && $STACK != sensor ]] && { echo "--stack must be 'full' or 'sensor'"; exit 1; }
# One data root for every window: nodes reached through logsetup resolve
# TINYNAV_DB_PATH at import, and without it they fall to /tinynav/tinynav_db --
# inside the checkout -- instead of the rig's data directory.
[[ -n $DB_PATH ]] && export TINYNAV_DB_PATH="$DB_PATH"
if [[ $STACK == sensor ]] && { [[ $WITH_MAP == 1 ]] || [[ -n $AUTO_SCENE ]]; }; then
  echo "--stack sensor conflicts with --map/--auto: pilot owns map_node and localization"; exit 1
fi
cd /tinynav
mkdir -p logs

# relaunch semantics: the previous rig must be fully gone before this one
# starts (gz, nodes, tmux session, dead DDS shared memory)
bash /tinynav/tool/simulator/kill_sim.sh || true

WORLD_NAME=$(grep -oP '(?<=<world name=")[^"]+' "$WORLD_SDF" | head -1)
[[ -z $WORLD_NAME ]] && { echo "no <world name> in $WORLD_SDF"; exit 1; }

# model:// resources shipped OUTSIDE the repo (factory_01 plant, kept in
# metaverse-source): parent dir resolves model://, model dir resolves the mesh
# URIs inside model.sdf. Override the root if the model moves elsewhere.
FACTORY_MODEL_ROOT=${FACTORY_MODEL_ROOT:-/workspace/dm/metaverse-source/converted}
if [[ -d $FACTORY_MODEL_ROOT/gz_model ]]; then
  export IGN_GAZEBO_RESOURCE_PATH="$FACTORY_MODEL_ROOT:$FACTORY_MODEL_ROOT/gz_model${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
fi

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

# Per-world spawn poses. Pose must be right at create time: a teleport after
# VIO init poisons the ISAM pose graph (factory's lekiwi start is the
# historical proof). Every model self-poses its chassis above a ground-level
# origin (lekiwi chassis 0.083, go2 trunk 0.27), so z=0 stands the robot on
# the world floor; factory adds its 0.26 deck.
case $WORLD_NAME in
  factory) TINYNAV_SPAWN_X=-5.71; TINYNAV_SPAWN_Y=2.68; TINYNAV_SPAWN_Z=0.28; TINYNAV_SPAWN_YAW=-1.57 ;;
  *)       TINYNAV_SPAWN_X=0;     TINYNAV_SPAWN_Y=0;     TINYNAV_SPAWN_Z=0;     TINYNAV_SPAWN_YAW=0 ;;
esac
if [[ $ROBOT == go2 ]]; then
  case $WORLD_NAME in
    factory) TINYNAV_SPAWN_X=0; TINYNAV_SPAWN_Y=0; TINYNAV_SPAWN_Z=0.31; TINYNAV_SPAWN_YAW=0 ;;
    *)       TINYNAV_SPAWN_X=0; TINYNAV_SPAWN_Y=0; TINYNAV_SPAWN_Z=0;     TINYNAV_SPAWN_YAW=0 ;;
  esac
fi
export TINYNAV_SPAWN_X TINYNAV_SPAWN_Y TINYNAV_SPAWN_Z TINYNAV_SPAWN_YAW
export TINYNAV_WORLD_NAME="$WORLD_NAME"
# model name for scene_runner's resets/targets
case $ROBOT in
  go2) TINYNAV_ROBOT_MODEL=go2 ;;
  *)   TINYNAV_ROBOT_MODEL=lekiwi ;;
esac
export TINYNAV_ROBOT_MODEL

# The gz robot's planning footprint must match its body: lekiwi is the 0.2m
# cylinder (the default go2 rectangle reads side obstacles within ~0.25m as
# footprint hits and stalls the planner in these aisles); go2 rigs run the
# quadruped so its own config is correct. Pre-set ROBOT_TYPE in the
# environment to override.
case $ROBOT in
  go2) export ROBOT_TYPE=${ROBOT_TYPE:-go2} ;;
  *)   export ROBOT_TYPE=${ROBOT_TYPE:-lekiwi} ;;
esac

# go2 spawns from URDF with a ros2_control system plugin inside the model;
# Fortress's SystemLoader ignores LD_LIBRARY_PATH and only searches this.
if [[ $ROBOT == go2 ]]; then
  export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib
fi

if [[ $ROBOT == go2 ]]; then
  # same sensor face as lekiwi, but /cmd_vel is consumed ROS-side by the gait
  # chain (gz has no diff-drive to receive it) and the control chain needs
  # /clock for ros2_control.
  msg_bridge_args="\
/camera/camera/infra1/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/infra2/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/color/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU \
/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock \
/world/$WORLD_NAME/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"
else
  msg_bridge_args="\
/camera/camera/infra1/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/infra2/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/color/image_raw@sensor_msgs/msg/Image@ignition.msgs.Image \
/camera/camera/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU \
/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
/world/$WORLD_NAME/pose/info@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"
fi

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
# the robot window is the ONLY spawn point; vars go IN the command string
# (new tmux windows inherit the server's environment, not this script's)
win robot "TINYNAV_WORLD_NAME='$WORLD_NAME' TINYNAV_SPAWN_X=$TINYNAV_SPAWN_X TINYNAV_SPAWN_Y=$TINYNAV_SPAWN_Y TINYNAV_SPAWN_Z=$TINYNAV_SPAWN_Z TINYNAV_SPAWN_YAW=$TINYNAV_SPAWN_YAW bash /tinynav/tool/simulator/robots/${ROBOT}/spawn.sh 2>&1 | tee logs/robot_spawn.log"
win bridge "ros2 run ros_gz_bridge parameter_bridge $msg_bridge_args 2>&1 | tee logs/bridge.log"
win caminfo "uv run python tool/simulator/gazebo_scene/camera_info_publisher.py 2>&1 | tee logs/caminfo.log"
PERCEPT_ARGS=""
if [[ $STACK == full && $WITH_MAP == 0 ]]; then
  # sim_gt_reloc maps goals (and the map frame) between the gz world and the
  # raw SLAM frame; perception emits its raw stream on ..._odometry_visual_raw.
  PERCEPT_ARGS="--ros-args -r /slam/odometry_visual:=/slam/odometry_visual_raw"
  win reloc "TINYNAV_ROBOT_MODEL=$TINYNAV_ROBOT_MODEL uv run python tool/simulator/gazebo_scene/sim_gt_reloc.py 2>&1 | tee logs/reloc.log"
fi
win percept "uv run python tinynav/core/perception_node.py $PERCEPT_ARGS 2>&1"
# trajectory follower on every rig: planning publishes /planning/trajectory_path,
# this node turns it into /cmd_vel (on the go2 the servo consumes it; lekiwi's
# diff-drive takes it directly)
win control "uv run python tinynav/platforms/simulator_control.py 2>&1 | tee logs/control.log"
if [[ $STACK == full ]]; then
PLAN_ARGS=""
if [[ $WITH_MAP == 0 ]]; then
  # no reloc-corrected odometry anymore: planning plans in perception's raw
  # SLAM frame and sim_gt_reloc only transforms gz-authored goals into it.
  # The sync input must be the per-frame visual odometry -- its stamps match
  # /slam/depth exactly; /slam/odometry (imu propagator) would starve the pair.
  PLAN_ARGS="--ros-args -r /slam/odometry_visual:=/slam/odometry_visual_raw"
fi
if [[ $ROBOT == go2 ]]; then
  # Real-rig go2 value: a near-ground cell needs 0.2 m of z-span to count
  # as obstacle (the class default 0.05 lets the trotting quadruped's
  # smeared floor hits through).
  PLAN_ARGS="$PLAN_ARGS -p min_wall_span_m:=0.2"
fi
win planning "uv run python tinynav/core/planning_node.py $PLAN_ARGS 2>&1 | tee logs/planning.log"
# keys publish /cmd_vel: on the go2 the servo consumes it, on lekiwi the
# diff-drive does
win teleop "uv run python tinynav/platforms/keyboard_teleop.py 2>&1 | tee logs/teleop.log"
  win rviz "rviz2 -d /tinynav/docs/vis.rviz 2>&1 | tee logs/rviz.log"
  if [[ $WITH_MAP == 1 ]]; then
    win map "uv run python tinynav/core/map_node.py --tinynav_map_path $MAP_DIR --tinynav_db_path $NAV_DB_DIR 2>&1 | tee logs/map.log"
  fi
  if [[ -n $AUTO_SCENE ]]; then
    AUTO_ARGS=""
    [[ $WITH_MAP == 1 ]] && AUTO_ARGS="--no-reloc"  # map_node is the authority
    win gz_scene "TINYNAV_ROBOT_MODEL=$TINYNAV_ROBOT_MODEL uv run python -u tool/simulator/gazebo_scene/scene_runner.py $AUTO_SCENE $AUTO_ARGS 2>&1 | tee logs/gz_scene.log"
  fi
fi

echo "session '$SESSION' up (STACK=$STACK, ROBOT=$ROBOT, WITH_MAP=$WITH_MAP, WORLD=$WORLD_SDF, AUTO=${AUTO_SCENE:-none}):"
tmux list-windows -t "$SESSION" -F '  #{window_index}:#{window_name}'
echo "attach: docker exec -t tinynav tmux attach -t $SESSION   (Ctrl+B D detach)"

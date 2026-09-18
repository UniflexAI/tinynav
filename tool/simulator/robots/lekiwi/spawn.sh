#!/usr/bin/env bash
# Spawn the lekiwi into the running gz world. Pose comes from the launcher's
# per-world table (TINYNAV_SPAWN_X/Y/Z/YAW); defaults put it at the origin.
set -u
command -v ros2 >/dev/null 2>&1 || source /opt/ros/humble/setup.bash
DIR=${TINYNAV_ROBOT_DIR:-/tinynav/tool/simulator/robots/lekiwi}
timeout 60 ros2 run ros_gz_sim create -world "${TINYNAV_WORLD_NAME:?TINYNAV_WORLD_NAME not set}" \
  -file "$DIR/lekiwi.sdf" -name lekiwi \
  -x "${TINYNAV_SPAWN_X:-0}" -y "${TINYNAV_SPAWN_Y:-0}" -z "${TINYNAV_SPAWN_Z:-0}" \
  -Y "${TINYNAV_SPAWN_YAW:-0}" && echo "[lekiwi] spawned"

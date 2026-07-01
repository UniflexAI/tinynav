#!/bin/bash
set -euo pipefail

# QR-odometry + EKF fusion pipeline.
#
# Requires tinynav_db/qrcode/tag_mappose.json to already exist
# (produced by run_qr_calibrate.sh). Launches looper_bridge_node + map_node
# for world->map TF, odom_node.py to detect the AprilTag board and publish
# /qr/odom, and ekf_odom_node.py to fuse /slam/odometry + /qr/odom into
# /slam/odometry_fused.
#
# Usage: run_qr_ekf.sh [map_path]

map_path="${1:-/tinynav/tinynav_db/map_guangzhou_office}"

if [ ! -f "tinynav_db/qrcode/tag_mappose.json" ] && [ ! -f "/tinynav/tinynav_db/qrcode/tag_mappose.json" ]; then
  echo "tag_mappose.json not found — run scripts/run_qr_calibrate.sh first." >&2
  exit 1
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 3 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/looper_bridge_node.py" C-m \; \
  select-pane -t 1 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path" C-m \; \
  select-pane -t 2 \; send-keys 'uv run python /tinynav/tool/qr_odom/odom_node.py' C-m \; \
  select-pane -t 3 \; send-keys 'uv run python /tinynav/tool/ekf_odom_node.py' C-m \; \
  select-pane -t 4 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

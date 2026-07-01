#!/bin/bash
set -euo pipefail

# One-shot AprilTag-board calibration.
#
# Launches looper_bridge_node + map_node so the robot relocalizes against the
# pre-built map, then leaves record_node.py staged (unstarted) in its own pane.
#
# Usage: run_qr_calibrate.sh [map_path]
#
# Steps:
#   1. Wait for map_node to relocalize (watch rviz / pane 1 logs for
#      "world -> map" TF / relocalization success).
#   2. Point the camera at the AprilTag board.
#   3. Switch to the last pane (record_node.py, already typed) and press Enter.
#      It writes tinynav_db/qrcode/tag_target.json and tag_mappose.json.

map_path="${1:-/tinynav/tinynav_db/map_guangzhou_office}"

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/looper_bridge_node.py" C-m \; \
  select-pane -t 1 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path" C-m \; \
  select-pane -t 2 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 3 \; send-keys 'uv run python /tinynav/tool/qr_odom/record_node.py'

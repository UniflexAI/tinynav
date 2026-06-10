#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PERSPECTIVE_FILE="$ROOT_DIR/docs/preview.perspective"
SESSION_NAME="tinynav_sensors_preview"

sensor_cmd="bash $ROOT_DIR/scripts/run_realsense_sensor.sh"
if ros2 node list 2>/dev/null | grep -qx "/insight_full"; then
  echo "Looper detected (/insight_full). Using compressed RGB republish for preview."
  sensor_cmd="ros2 run image_transport republish compressed raw --ros-args -r in/compressed:=/camera/camera/color/image_rect_raw/compressed -r out:=/camera/camera/color/image_raw"
fi

tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true
tmux new-session -d -s "$SESSION_NAME" \; \
  split-window -v \; \
  select-pane -t 0 \; send-keys "$sensor_cmd" C-m \; \
  select-pane -t 1 \; send-keys "rqt --perspective-file $PERSPECTIVE_FILE" C-m

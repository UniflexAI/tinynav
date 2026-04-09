#!/bin/bash
set -euo pipefail

tmux new-session \; \
  split-window -v \; \
  select-pane -t 0 \; send-keys "bash /tinynav/scripts/run_realsense_sensor.sh" C-m \; \
  select-pane -t 1 \; send-keys "sleep 3; bash /tinynav/scripts/run_rosbag_record.sh" C-m

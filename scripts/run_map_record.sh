#!/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

tmux new-session \; \
  split-window -v \; \
  select-pane -t 0 \; send-keys "bash '${script_dir}/run_realsense_sensor.sh'" C-m \; \
  select-pane -t 1 \; send-keys "sleep 3; bash '${script_dir}/run_rosbag_record.sh'" C-m

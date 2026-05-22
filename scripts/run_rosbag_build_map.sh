#!/bin/bash
set -euo pipefail

rosbag_path=/tinynav/tinynav_db/rosbags/bag_2026_05_20_19_22_13
map_save_path=/tinynav/tinynav_db/maps/map_2026_05_21_10_51_10

mode="${1:-looper_direct}" # looper_direct | perception

if [[ "${mode}" == "looper_direct" ]]; then
  source_cmd='uv run python /tinynav/tool/looper_bridge_node.py'
elif [[ "${mode}" == "perception" ]]; then
  source_cmd='uv run python /tinynav/tinynav/core/perception_node.py'
else
  echo "Usage: $0 [looper_direct|perception]"
  exit 1
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 1 \; send-keys "uv run python /tinynav/tinynav/core/build_map_node.py --map_save_path $map_save_path --bag_file $rosbag_path" C-m \; \
  select-pane -t 2 \; send-keys "${source_cmd}" C-m \; \

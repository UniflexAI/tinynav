#!/bin/bash
set -euo pipefail

rosbag_path=$(uv run hf download --repo-type dataset --cache-dir /tinynav UniflexAI/rosbag2_go2_looper)
map_save_path=/tinynav/output/map_go2_looper
tinynav_setup='source /opt/ros/humble/setup.bash && source /3rdparty/message_filters_ws/install/local_setup.bash &&'

mode="${1:-perception}" # looper_direct | perception

if [[ "${mode}" == "looper_direct" ]]; then
  source_cmd="${tinynav_setup} uv run python /tinynav/tool/looper_bridge_node.py"
elif [[ "${mode}" == "perception" ]]; then
  source_cmd="${tinynav_setup} uv run python /tinynav/tinynav/core/perception_node.py"
else
  echo "Usage: $0 [looper_direct|perception]"
  exit 1
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 1 \; send-keys "${tinynav_setup} uv run python /tinynav/tinynav/core/build_map_node.py --map_save_path $map_save_path --bag_file $rosbag_path" C-m \; \
  select-pane -t 2 \; send-keys "${source_cmd}" C-m \; \
  select-pane -t 3 \; send-keys "${tinynav_setup} ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz" C-m

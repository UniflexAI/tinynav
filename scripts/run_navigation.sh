#!/bin/bash

export ROS_DOMAIN_ID=18
map_path="device_sync/maps/map_2026_05_14_21_45_55/"

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 3 \; split-window -v \; \
  select-pane -t 4 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path --loop-closure-use-bow" C-m \; \
  select-pane -t 3 \; send-keys "uv run python tool/poi_editor.py --tinynav-map_path $map_path" C-m \; \
  select-pane -t 4 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 5 \; send-keys "sleep 3 && uv run python /tinynav/tool/pub_pois.py --tinynav_map_path $map_path" C-m


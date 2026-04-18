#!/bin/bash

map_path="/tinynav/output/map_hangzhou_juluo_0415"

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/looper_bridge_node.py" C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path" C-m \; \
  select-pane -t 3 \; send-keys "uv run python /tinynav/tinynav/platforms/cmd_vel_control.py" C-m \; \
  select-pane -t 4 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

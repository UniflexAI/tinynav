#!/bin/bash

map_path="PATH_TO_MAP"

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys 'uv run python /tinynav/tinynav/platforms/unitree_control.py' C-m \; \
  select-pane -t 3 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path" C-m \; \
  select-pane -t 4 \; send-keys "uv run python /tinynav/platforms/cmd_vel_control.py" C-m \; \
  select-pane -t 5 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 6 \; send-keys 'bash' C-m


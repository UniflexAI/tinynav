#!/bin/bash
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys "uv run python /tinynav/tinynav/core/build_map_node.py --map_save_path /tinynav/tinynav_db/map --bag_file /tinynav/tinynav_db/bag" C-m \; \
  select-pane -t 3 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

#!/bin/bash
rosbag_path=$(uv run huggingface-cli download --repo-type dataset --cache-dir /tinynav UniflexAI/rosbag2_go2_exposure_1k)

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/map_node.py --mapping True' C-m \; \
  select-pane -t 2 \; send-keys "ros2 bag play $rosbag_path" C-m \; \
  select-pane -t 3 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

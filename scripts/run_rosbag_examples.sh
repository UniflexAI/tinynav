#!/bin/bash
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys 'ros2 bag play $(uv run huggingface-cli download --repo-type dataset --cache-dir /tinynav UniflexAI/rosbag_d435i_go2)' C-m \; \
  select-pane -t 3 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 4 \; send-keys 'bash' C-m

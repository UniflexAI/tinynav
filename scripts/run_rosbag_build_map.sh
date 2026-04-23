#!/bin/bash

SESSION_NAME="tinynav_build_map"
rosbag_path=/tinynav/rosbag2_go2_simulation
map_save_path=/tinynav/output/map_go2_simulation

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? -eq 0 ]; then
    echo "Session $SESSION_NAME already exists. Attaching..."
    tmux attach -t $SESSION_NAME
    exit 0
fi

tmux new-session -s $SESSION_NAME \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys "uv run python /tinynav/tinynav/core/build_map_node.py --map_save_path $map_save_path --bag_file $rosbag_path" C-m \; \
  select-pane -t 3 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

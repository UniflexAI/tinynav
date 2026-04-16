#!/bin/bash
export ROS_DOMAIN_ID=19

rosbag_path=/tinynav/tinynav_db/rosbags/map_record_20260415_011314
map_save_path=/tinynav/tinynav_db/map
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'uv run python /tinynav/tinynav/core/perception_node.py' C-m \; \
  select-pane -t 1 \; send-keys "uv run python /tinynav/tinynav/core/build_map_node.py --map_save_path $map_save_path --bag_file $rosbag_path" C-m \; 

#!/bin/bash

map_path="${TINYNAV_MAP_PATH:-/tinynav/output/map_hangzhou_juluo_0415}"
pose_topic="${LOOPER_POSE_TOPIC:-/insight/vio_pose}"
depth_topic="${LOOPER_DEPTH_TOPIC:-/camera/camera/depth/image_rect_raw}"
image_topic="${LOOPER_IMAGE_TOPIC:-/camera/camera/infra1/image_rect_raw}"
camera_info_topic="${LOOPER_CAMERA_INFO_TOPIC:-/camera/camera/infra1/camera_info}"

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/looper_bridge_node.py --pose-topic $pose_topic --depth-topic $depth_topic --image-topic $image_topic --camera-info-topic $camera_info_topic" C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys "uv run python /tinynav/tinynav/core/map_node.py --tinynav_map_path $map_path" C-m \; \
  select-pane -t 3 \; send-keys "uv run python /tinynav/tinynav/platforms/cmd_vel_control.py" C-m \; \
  select-pane -t 4 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

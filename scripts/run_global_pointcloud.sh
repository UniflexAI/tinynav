#!/bin/bash

POINTCLOUD_MODE=${POINTCLOUD_MODE:-color}
POSE_TOPIC=${POSE_TOPIC:-/insight/vio_pose}

tmux new-session \; \
  split-window -h \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/global_pointcloud_publisher.py --image-mode $POINTCLOUD_MODE --pose-topic $POSE_TOPIC" C-m \; \
  select-pane -t 1 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

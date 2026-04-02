#!/bin/bash

POINTCLOUD_MODE=${POINTCLOUD_MODE:-color}
OUTPUT_TOPIC=${OUTPUT_TOPIC:-/global_pointcloud}
GLOBAL_RADIUS=${GLOBAL_RADIUS:-100.0}
VOXEL_SIZE=${VOXEL_SIZE:-0.05}

COMMON_ARGS="--output-topic $OUTPUT_TOPIC --global-radius $GLOBAL_RADIUS --voxel-size $VOXEL_SIZE"
GRAY_ARGS='--image-mode grayscale --gray-image-topic /camera/camera/infra1/image_rect_raw --camera-info-topic /camera/camera/infra1/camera_info'
COLOR_ARGS='--image-mode color --color-image-topic /camera/camera/color/image_rect_raw/compressed --color-camera-info-topic /camera/camera/color/camera_info'

if [ "$POINTCLOUD_MODE" = "grayscale" ]; then
  PY_ARGS="$COMMON_ARGS $GRAY_ARGS"
else
  PY_ARGS="$COMMON_ARGS $COLOR_ARGS"
fi

tmux new-session \; \
  split-window -h \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/global_pointcloud_publisher.py $PY_ARGS" C-m \; \
  select-pane -t 1 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m

#!/bin/bash

set -e

# Create a tmux session with 4 panes
# 1. Run the foxglove video encoder
# 2. Run the foxglove bridge
# 3. Monitor the bandwidth of the compressed topic
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; send-keys 'ros2 run foxglove_bridge foxglove_bridge --port 8765 --topic-whitelist /camera/camera/infra1/image_rect_raw_repub,/camera/camera/infra1/camera_info' C-m \; \
  select-pane -t 1 \; send-keys 'ros2 run image_transport republish raw foxglove --ros-args --remap in:=/camera/camera/infra1/image_rect_raw -r out/foxglove:=/camera/camera/infra1/image_rect_raw_repub -p out.foxglove.qmax:=60 -p out.foxglove.bit_rate:=8000000' C-m \; \
  select-pane -t 2 \; send-keys 'ros2 topic bw /camera/camera/infra1/image_rect_raw_repub' C-m 

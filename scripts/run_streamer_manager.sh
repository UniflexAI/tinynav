#!/bin/bash
SESSION_NAME=${TMUX_SESSION_NAME:-streamer}
tmux new-session -s $SESSION_NAME \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 -p topic_whitelist:="[/camera/camera/color/image_raw_repub,/camera/camera/color/camera_info, /slam/disparity_vis, /tf, /cmd_vel, /service/command, /service/state, slam/odometry, /planning/trajectory_path, /planning/occupied_voxels, /battery, /mapping/pointcloud_markers, /mapping/global_plan, /mapping/poi]"' C-m \; \
  select-pane -t 1 \; send-keys 'ros2 run image_transport republish raw foxglove --ros-args --remap in:=/camera/camera/color/image_raw -r out/foxglove:=/camera/camera/color/image_raw_repub -p out.foxglove.qmax:=60 -p out.foxglove.bit_rate:=8000000' C-m \; \
  select-pane -t 2 \; send-keys 'uv run python /tinynav/tinynav/platforms/unitree_control.py' C-m \; \
  select-pane -t 3 \; send-keys 'uv run python /tinynav/tool/ros2_node_manager.py' C-m 


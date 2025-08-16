#!/bin/bash

msg_bridge_args="/camera/camera/infra1/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image /camera/camera/infra2/image_rect_raw@sensor_msgs/msg/Image@ignition.msgs.Image /camera/camera/accel/sample@sensor_msgs/msg/Imu@ignition.msgs.IMU /cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist"

set -e
# Create tmux session with multiple panes
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 2 \; split-window -v \; \
  select-pane -t 4 \; split-window -v \; \
  select-pane -t 0 \; send-keys 'echo "Starting Gazebo simulator..." && ign gazebo -v 4 tool/simulator/robot_scene.sdf 2>&1 | tee logs/gazebo.log' C-m \; \
  select-pane -t 1 \; send-keys "ros2 run ros_gz_bridge parameter_bridge $msg_bridge_args" C-m \; \
  select-pane -t 2 \; send-keys "uv run python tool/simulator/camera_info_publisher.py" C-m \; \
  select-pane -t 3 \; send-keys "uv run python tinynav/core/perception_node.py" C-m \; \
  select-pane -t 4 \; send-keys "uv run python tinynav/core/planning_node.py" C-m \; \
  select-pane -t 5 \; send-keys "uv run python tinynav/platforms/simulator_control.py" C-m \; \
  select-pane -t 6 \; send-keys "rviz2 -d /tinynav/docs/vis.rviz" C-m


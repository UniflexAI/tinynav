#!/bin/bash
# Stair mode with the Looper camera: stair_node replaces map_node as the target source.
# Last pane has the stair command typed but not sent; edit up/down and press Enter.
# cmd_vel_control only moves while /nav/active is true (normally set by the app).

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 3 \; split-window -v \; \
  select-pane -t 4 \; split-window -v \; \
  select-pane -t 0 \; send-keys "uv run python /tinynav/tool/looper_bridge_node.py" C-m \; \
  select-pane -t 1 \; send-keys 'uv run python /tinynav/tinynav/core/planning_node.py' C-m \; \
  select-pane -t 2 \; send-keys "uv run python /tinynav/tinynav/core/stair_node.py" C-m \; \
  select-pane -t 3 \; send-keys "uv run python /tinynav/tinynav/platforms/cmd_vel_control.py" C-m \; \
  select-pane -t 4 \; send-keys 'ros2 run rviz2 rviz2 -d /tinynav/docs/vis.rviz' C-m \; \
  select-pane -t 5 \; send-keys "ros2 topic pub --once /stair/cmd std_msgs/msg/String '{data: down}'"

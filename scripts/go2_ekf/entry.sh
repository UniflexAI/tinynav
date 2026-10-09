#!/bin/bash
# Inside the go2-ekf container: leg odometry, VIO twist and the EKF; if any of them exits, stop all (docker restarts nothing).
source /opt/ros/humble/setup.bash
cd /tinynav   # plain /opt/venv python: uv run would re-sync the image venv against the repo lock on every start
python3 tinynav/platforms/go2_leg_odom_node.py & P1=$!
python3 tinynav/platforms/go2_vio_twist_node.py & P2=$!
ros2 run robot_localization ekf_node --ros-args -r __node:=go2_ekf --params-file scripts/go2_ekf/ekf.yaml \
    -r odometry/filtered:=/go2/ekf/odometry & P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null' TERM INT
wait -n
kill $P1 $P2 $P3 2>/dev/null; wait

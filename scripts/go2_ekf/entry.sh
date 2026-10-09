#!/bin/bash
# Inside the go2-ekf container: leg odometry, VIO twist and the EKF. If any of them exits, all stop and docker restarts the container.
source /opt/ros/humble/setup.bash
cd /tinynav   # plain /opt/venv python: uv run would re-sync the image venv against the repo lock on every start
# The Jetson can boot without the dog's Ethernet (unplugged, dog off); the Unitree SDK exits at once if it is missing
until [ "$(cat /sys/class/net/enP8p1s0/operstate 2>/dev/null)" = up ]; do sleep 2; done
python3 tinynav/platforms/go2_leg_odom_node.py & P1=$!
python3 tinynav/platforms/go2_vio_twist_node.py & P2=$!
ros2 run robot_localization ekf_node --ros-args -r __node:=go2_ekf --params-file scripts/go2_ekf/ekf.yaml \
    -r odometry/filtered:=/go2/ekf/odometry & P3=$!
trap 'kill $P1 $P2 $P3 2>/dev/null; exit 0' TERM INT
wait -n
kill $P1 $P2 $P3 2>/dev/null; wait
exit 1

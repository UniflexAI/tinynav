#!/bin/bash
# Inside the go2-ekf container: leg odometry, VIO twist and the EKF. The leg node exits when lowstate stops and is
# restarted on its own, so the EKF keeps its origin; if the VIO node or the EKF exits, all stop and docker restarts us.
source /opt/ros/humble/setup.bash
cd /tinynav   # plain /opt/venv python: uv run would re-sync the image venv against the repo lock on every start
NIC=/sys/class/net/enP8p1s0
nic() { echo "enP8p1s0 operstate=$(cat $NIC/operstate 2>/dev/null) carrier=$(cat $NIC/carrier 2>/dev/null) rx_packets=$(cat $NIC/statistics/rx_packets 2>/dev/null)"; }
legs() {
    while true; do
        # The Jetson can boot without the dog's Ethernet (unplugged, dog off); the Unitree SDK exits at once if it is missing
        until [ "$(cat $NIC/operstate 2>/dev/null)" = up ]; do sleep 2; done
        python3 tinynav/platforms/go2_leg_odom_node.py
        # rx_packets still rising = the link is alive and the dog went quiet; flat = the link itself died (10-10 17:49)
        echo "go2_leg_odom exited ($?): $(nic)"; sleep 1; echo "go2_leg_odom 1 s later: $(nic)"
    done
}
legs & P1=$!
python3 tinynav/platforms/go2_vio_twist_node.py & P2=$!
ros2 run robot_localization ekf_node --ros-args -r __node:=go2_ekf --params-file scripts/go2_ekf/ekf.yaml \
    -r odometry/filtered:=/go2/ekf/odometry & P3=$!
stop() { kill $P1 $P2 $P3 2>/dev/null; pkill -f go2_leg_odom_node.py; }
trap 'stop; exit 0' TERM INT
wait -n
stop; wait
exit 1

#!/bin/bash
# Inside the go2-ekf container: leg odometry, VIO twist and the EKF. The leg node exits when lowstate stops and is
# restarted on its own, so the EKF keeps its origin; if the VIO node or the EKF exits, all stop and docker restarts us.
source /opt/ros/humble/setup.bash
cd /tinynav   # plain /opt/venv python: uv run would re-sync the image venv against the repo lock on every start
NIC=/sys/class/net/enP8p1s0
nic() { echo "enP8p1s0 operstate=$(cat $NIC/operstate 2>/dev/null) carrier=$(cat $NIC/carrier 2>/dev/null) rx_packets=$(cat $NIC/statistics/rx_packets 2>/dev/null)"; }
rx() { cat $NIC/statistics/rx_packets 2>/dev/null || echo 0; }
# On 10-10 the Jetson's Realtek NIC (r8168, EEE active) stopped receiving five times with the carrier up; the dog was
# fine and `ip link set down/up` brought ~1800 packets/s straight back. EEE is the usual suspect; off costs < 0.5 W.
eee_off() {
    ethtool --show-eee enP8p1s0 2>/dev/null | grep -q 'EEE status: enabled' && ethtool --set-eee enP8p1s0 eee off && echo "EEE turned off on enP8p1s0"
}
legs() {
    last_reset=0
    while true; do
        # The Jetson can boot without the dog's Ethernet (unplugged, dog off); the Unitree SDK exits at once if it is missing
        until [ "$(cat $NIC/operstate 2>/dev/null)" = up ]; do sleep 2; done
        eee_off
        python3 tinynav/platforms/go2_leg_odom_node.py; rc=$?
        r0=$(rx); echo "go2_leg_odom exited ($rc): $(nic)"; sleep 1; r1=$(rx); echo "go2_leg_odom 1 s later: $(nic)"
        # carrier up yet (almost) nothing arriving: the receive side hung, not the dog (it sends ~1800 packets/s)
        if [ "$(cat $NIC/carrier 2>/dev/null)" = 1 ] && [ $((r1 - r0)) -lt 50 ] && [ $(($(date +%s) - last_reset)) -gt 30 ]; then
            echo "enP8p1s0 receives nothing with the carrier up: resetting the link"
            ip link set enP8p1s0 down; sleep 1; ip link set enP8p1s0 up; last_reset=$(date +%s)
        fi
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

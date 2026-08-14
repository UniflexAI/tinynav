#!/bin/bash
# Records VIO, LIO, and the ekf_odom_node fusion output for offline debugging.
#
# /lidar/odom (published by super_lio on the A2 onboard PC) has repeatedly failed to
# actually deliver data to ad-hoc subscribers created on the Jetson side, even though
# `ros2 topic list`/`topic info` show it as discovered and matched (see the "beijing
# lidar/odom cross-host subscribe" investigation). Recording it locally on the A2 board
# is the only way that has reliably captured data. So this script runs two independent
# `ros2 bag record` processes -- one here (VIO + the ekf fusion output), one over SSH on
# the A2 board (LIO) -- rather than trying to subscribe to everything from one host.
#
# Usage (run inside the tinynav-dev container, on the Jetson):
#   A2_PASSWORD=... ./tool/record_vio_lio_fusion_debug.sh [--output DIR] [--duration SECONDS]
#
# Without --duration, recording runs until Ctrl-C. Either way, both recorders are
# stopped together and the A2-side bag is copied back next to the local one.
set -uo pipefail

A2_HOST="${A2_HOST:-192.168.123.162}"
A2_USER="${A2_USER:-unitree}"
A2_PASSWORD="${A2_PASSWORD:?Set A2_PASSWORD to the SSH password for the A2 onboard PC}"
A2_CYCLONEDDS_URI="${A2_CYCLONEDDS_URI:-file:///home/unitree/slam_config/cyclonedds_go2_B2_ws/cyclonedds.xml}"
JETSON_CYCLONEDDS_URI="${JETSON_CYCLONEDDS_URI:-/tinynav/cyclonedds_jetson.xml}"

LOCAL_TOPICS=(/slam/odometry /slam/odometry_100hz /slam/odometry_fused /slam/odometry_fused_100hz /tf_static)
REMOTE_TOPICS=(/lidar/odom /lidar/points /lidar/imu)

output_dir=""
duration=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o) output_dir="$2"; shift 2 ;;
        --duration|-d) duration="$2"; shift 2 ;;
        *) echo "Usage: $0 [--output DIR] [--duration SECONDS]" >&2; exit 1 ;;
    esac
done

timestamp="$(date +%Y%m%d_%H%M%S)"
if [ -z "$output_dir" ]; then
    xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
    output_dir="${xdg_data_home}/tinynav/rosbags/vio_lio_fusion_debug_${timestamp}"
fi
mkdir -p "$output_dir"

local_bag="${output_dir}/local_vio_fusion"
local_log="${output_dir}/local_record.log"
remote_bag_dir="/tmp/lio_debug_bag_${timestamp}"
remote_log="/tmp/lio_debug_bag_${timestamp}.log"

echo "[record] local bag  -> ${local_bag}  (topics: ${LOCAL_TOPICS[*]})"
echo "[record] remote bag -> ${A2_USER}@${A2_HOST}:${remote_bag_dir}  (topics: ${REMOTE_TOPICS[*]})"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="$JETSON_CYCLONEDDS_URI"
source /opt/ros/humble/setup.bash

ros2 bag record -o "$local_bag" "${LOCAL_TOPICS[@]}" > "$local_log" 2>&1 &
local_pid=$!

remote_pid="$(sshpass -p "$A2_PASSWORD" ssh -o StrictHostKeyChecking=no "${A2_USER}@${A2_HOST}" "
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=${A2_CYCLONEDDS_URI}
    source /opt/ros/humble/setup.bash
    nohup ros2 bag record -o ${remote_bag_dir} ${REMOTE_TOPICS[*]} > ${remote_log} 2>&1 < /dev/null &
    echo \$!
    disown
")"
remote_pid="$(echo "$remote_pid" | tr -d '[:space:]')"

if [ -z "$remote_pid" ]; then
    echo "[record] FAILED to start the remote (A2) recorder over SSH -- killing local recorder and aborting." >&2
    kill -INT "$local_pid" 2>/dev/null
    exit 1
fi
echo "[record] local_pid=${local_pid}  remote_pid=${remote_pid}"

sleep 4

echo "[record] --- verifying every requested topic actually got subscribed ---"
missing=0
for t in "${LOCAL_TOPICS[@]}"; do
    if ! grep -qF "Subscribed to topic '${t}'" "$local_log"; then
        echo "[record] WARNING: local recorder never subscribed to ${t} (not publishing right now?)" >&2
        missing=1
    fi
done
remote_log_contents="$(sshpass -p "$A2_PASSWORD" ssh -o StrictHostKeyChecking=no "${A2_USER}@${A2_HOST}" "cat ${remote_log} 2>/dev/null")"
for t in "${REMOTE_TOPICS[@]}"; do
    if ! echo "$remote_log_contents" | grep -qF "Subscribed to topic '${t}'"; then
        echo "[record] WARNING: remote (A2) recorder never subscribed to ${t} (not publishing right now?)" >&2
        missing=1
    fi
done
if [ "$missing" -eq 0 ]; then
    echo "[record] all requested topics confirmed subscribed on both sides."
else
    echo "[record] some topics are missing -- see warnings above. Recording continues; check before trusting the bag." >&2
fi

stopped=0
stop_both() {
    if [ "$stopped" -eq 1 ]; then return; fi
    stopped=1
    echo ""
    echo "[record] stopping both recorders..."
    kill -INT "$local_pid" 2>/dev/null
    sshpass -p "$A2_PASSWORD" ssh -o StrictHostKeyChecking=no "${A2_USER}@${A2_HOST}" "kill -INT ${remote_pid} 2>/dev/null"
    wait "$local_pid" 2>/dev/null
    sleep 2   # give the remote recorder time to flush its sqlite db before we scp it
    echo "[record] copying remote bag back from ${A2_HOST}..."
    sshpass -p "$A2_PASSWORD" scp -o StrictHostKeyChecking=no -r "${A2_USER}@${A2_HOST}:${remote_bag_dir}" "${output_dir}/remote_lidar_odom"
    echo ""
    echo "[record] done."
    echo "  local  bag (VIO + ekf fusion, on this host): ${local_bag}"
    echo "  remote bag (LIO, copied from A2):            ${output_dir}/remote_lidar_odom"
}
trap stop_both INT TERM

if [ -n "$duration" ]; then
    echo "[record] recording for ${duration}s..."
    sleep "$duration"
    stop_both
else
    echo "[record] recording -- press Ctrl-C when done."
    wait "$local_pid"
    stop_both
fi

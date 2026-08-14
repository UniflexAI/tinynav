#!/bin/bash
# Records VIO, LIO, and the ekf_odom_node fusion output for offline debugging, all in
# one `ros2 bag record` on the Jetson.
#
# /lidar/odom (published by super_lio on the A2 onboard PC) used to be undeliverable to
# subscribers on the Jetson even though `ros2 topic list`/`topic info` showed it as
# discovered and matched: super-lio.service was missing the RMW_IMPLEMENTATION/
# CYCLONEDDS_URI env vars (present on hesai-driver.service, its sibling unit), so it ran
# under the default FastDDS while everything else runs CycloneDDS -- cross-vendor RTPS
# discovery partially works (enough to show up in topic list) but data delivery does
# not. Fixed 2026-08-14 by adding those two Environment= lines to
# /etc/systemd/system/super-lio.service on the A2 board. If /lidar/odom silently stops
# showing up in this recording again, check that fix hasn't regressed (systemd unit
# edits don't survive a reflash/reimage of the A2 board).
#
# Usage (run inside the tinynav-dev container, on the Jetson):
#   ./tool/record_vio_lio_fusion_debug.sh [--output DIR] [--duration SECONDS]
#
# Without --duration, recording runs until Ctrl-C.
set -o pipefail  # not -u: /opt/ros/humble/setup.bash itself references unset vars

JETSON_CYCLONEDDS_URI="${JETSON_CYCLONEDDS_URI:-/tinynav/cyclonedds_jetson.xml}"

TOPICS=(/slam/odometry /slam/odometry_100hz /slam/odometry_fused /slam/odometry_fused_100hz \
        /lidar/odom /lidar/points /lidar/imu /tf_static)

output_dir=""
duration=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o) output_dir="$2"; shift 2 ;;
        --duration|-d) duration="$2"; shift 2 ;;
        *) echo "Usage: $0 [--output DIR] [--duration SECONDS]" >&2; exit 1 ;;
    esac
done

if [ -z "$output_dir" ]; then
    xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    output_dir="${xdg_data_home}/tinynav/rosbags/vio_lio_fusion_debug_${timestamp}"
fi
mkdir -p "$(dirname "$output_dir")"

bag="${output_dir}"
log="${output_dir}.log"

echo "[record] bag -> ${bag}  (topics: ${TOPICS[*]})"

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="$JETSON_CYCLONEDDS_URI"
source /opt/ros/humble/setup.bash

ros2 bag record -o "$bag" "${TOPICS[@]}" > "$log" 2>&1 &
pid=$!
echo "[record] pid=${pid}"

sleep 4

echo "[record] --- verifying every requested topic actually got subscribed ---"
missing=0
for t in "${TOPICS[@]}"; do
    if ! grep -qF "Subscribed to topic '${t}'" "$log"; then
        echo "[record] WARNING: never subscribed to ${t} (not publishing right now?)" >&2
        missing=1
    fi
done
if [ "$missing" -eq 0 ]; then
    echo "[record] all requested topics confirmed subscribed."
else
    echo "[record] some topics are missing -- see warnings above. Recording continues; check before trusting the bag." >&2
fi

stopped=0
stop_it() {
    if [ "$stopped" -eq 1 ]; then return; fi
    stopped=1
    echo ""
    echo "[record] stopping..."
    kill -INT "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "[record] done. bag: ${bag}"
}
trap stop_it INT TERM

if [ -n "$duration" ]; then
    echo "[record] recording for ${duration}s..."
    sleep "$duration"
    stop_it
else
    echo "[record] recording -- press Ctrl-C when done."
    wait "$pid"
    stop_it
fi

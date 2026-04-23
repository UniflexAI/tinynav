#!/bin/bash
set -euo pipefail

xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
record_root="/tinynav/rosbags"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${record_root}/bias_record_${timestamp}"

mkdir -p "${record_root}"

ros2 bag record \
    --output "${output_dir}" \
    --max-cache-size 2147483648 \
    /slam/imu_bias_accel \
    /slam/imu_bias_gyro \
    /camera/camera/imu \
    /slam/odometry \
    /slam/odometry_100hz

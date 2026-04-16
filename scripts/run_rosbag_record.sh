#!/bin/bash
set -euo pipefail

xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
record_root="/tinynav/tinynav_db/rosbags"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${record_root}/map_record_${timestamp}"

mkdir -p "${record_root}"

ros2 bag record \
    --output "${output_dir}" \
    --max-cache-size 2147483648 \
    /camera/camera/infra1/camera_info \
    /camera/camera/infra1/image_rect_raw \
    /camera/camera/infra1/metadata \
    /camera/camera/infra2/camera_info \
    /camera/camera/infra2/image_rect_raw \
    /camera/camera/infra2/metadata \
    /camera/camera/extrinsics/depth_to_infra1 \
    /camera/camera/extrinsics/depth_to_infra2 \
    /camera/camera/imu \
    /camera/camera/color/image_raw \
    /camera/camera/color/camera_info \
    /tf \
    /camera/camera/color/image_rect_raw/compressed \
    /tf_static

#!/bin/bash
set -euo pipefail

# Usage: run_rosbag_record.sh [--output DIR]
#   If --output is not given, a timestamped dir is created under XDG_DATA_HOME/tinynav/rosbags.

output_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o) output_dir="$2"; shift 2 ;;
        *) echo "Usage: $0 [--output DIR]" >&2; exit 1 ;;
    esac
done

if [ -z "$output_dir" ]; then
    xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
    record_root="${xdg_data_home}/tinynav/rosbags"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    output_dir="${record_root}/map_record_${timestamp}"
    mkdir -p "${record_root}"
else
    mkdir -p "$(dirname "$output_dir")"
fi

ros2 bag record \
    --output "${output_dir}" \
    --max-cache-size 2147483648 \
    /camera/camera/infra1/camera_info \
    /camera/camera/infra1/image_rect_raw \
    /camera/camera/infra1/metadata \
    /camera/camera/infra2/camera_info \
    /camera/camera/infra2/image_rect_raw \
    /camera/camera/infra2/metadata \
    /camera/camera/depth/image_rect_raw \
    /camera/camera/extrinsics/depth_to_infra1 \
    /camera/camera/extrinsics/depth_to_infra2 \
    /camera/camera/imu \
    /camera/camera/color/image_raw \
    /camera/camera/color/camera_info \
    /camera/camera/color/image_rect_raw/compressed \
    /insight/vio_20hz \
    /tf_static

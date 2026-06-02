#!/bin/bash
set -euo pipefail

# Usage: run_planning_debug_record.sh [--output DIR]
#   Records the minimum planning_node.py input topics for offline replay/debug.

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
    output_dir="${record_root}/planning_debug_${timestamp}"
    mkdir -p "${record_root}"
else
    mkdir -p "$(dirname "$output_dir")"
fi

ros2 bag record \
    --output "${output_dir}" \
    --max-cache-size 2147483648 \
    /slam/depth \
    /slam/odometry \
    /slam/odometry_visual \
    /planning/trajectory_path \
    /cmd_vel \
    /camera/camera/infra2/camera_info \
    /control/target_pose \
    /mapping/poi_change

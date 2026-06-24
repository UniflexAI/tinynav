#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_rosbag> [output_dir] [extra tool args...]"
  exit 1
fi

input_bag="$1"
shift
output_dir="${1:-/tinynav/output/perception_vio_benchmark_$(date +%Y%m%d_%H%M%S)}"
if [[ $# -gt 0 ]]; then
  shift
fi

uv run python /tinynav/tool/run_perception_trajectory_benchmark.py run \
  --input_bag "${input_bag}" \
  --output_dir "${output_dir}" \
  "$@"

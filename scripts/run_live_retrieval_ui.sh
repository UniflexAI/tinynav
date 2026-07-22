#!/bin/bash
set -euo pipefail

# Live top-1 VLAD retrieval viewer: replays a query rosbag and shows the
# current top-1 matching keyframe from a pre-built map in a browser.
#
# Usage: run_live_retrieval_ui.sh [map_path] [query_bag] [port]

map_path="${1:-/tinynav/output/map_day_20260716}"
query_bag="${2:-/tinynav/dataset/202601718/rosbags/bag_2026_07_17_20_32_32}"
port="${3:-8642}"

uv run python /tinynav/tool/live_retrieval_ui.py \
  --tinynav_map_path "$map_path" \
  --query_bag "$query_bag" \
  --port "$port"

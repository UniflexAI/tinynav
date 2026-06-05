#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <bag_path> <map_path> [output_jsonl]"
  echo "Example:"
  echo "  $0 /mnt/nas/.../debug_2026_05_23_15_55_46 /mnt/nas/.../map_2026_05_23_15_00_45"
  exit 1
fi

BAG_PATH="$1"
MAP_PATH="$2"
OUT_JSONL="${3:-/tinynav/tinynav_temp/retrieval_from_bag.jsonl}"
DEBUG_DIR="/tinynav/tinynav_temp/retrieval_from_bag_debug"

mkdir -p "$(dirname "$OUT_JSONL")" "$DEBUG_DIR"

cd /tinynav
python3 tool/retrieve_from_rosbag_map.py \
  --bag_path "$BAG_PATH" \
  --map_path "$MAP_PATH" \
  --topic /camera/camera/infra1/image_rect_raw \
  --topk 3 \
  --threshold 0.75 \
  --every_n 5 \
  --out_jsonl "$OUT_JSONL" \
  --save_debug_dir "$DEBUG_DIR"

echo "Saved retrieval jsonl: $OUT_JSONL"
echo "Saved debug images: $DEBUG_DIR"
echo "Saved review sessions under: /tinynav/tinynav_temp/retrieval_from_bag_review"

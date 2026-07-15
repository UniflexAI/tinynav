#!/usr/bin/env bash
set -euo pipefail

# VLAD retrieval test script
# Usage: ./scripts/run_vlad_retrieval_test.sh <bag_path> <map_path>

BAG_PATH="${1:?Usage: $0 <bag_path> <map_path>}"
MAP_PATH="${2:?Usage: $0 <bag_path> <map_path>}"
OUT_JSONL="/tinynav/tinynav_temp/vlad_retrieval.jsonl"
DEBUG_DIR="/tinynav/tinynav_temp/vlad_retrieval_debug"

mkdir -p "$(dirname "$OUT_JSONL")" "$DEBUG_DIR"

echo "=== Step 1: Generate VLAD for map (if not exists) ==="
if [[ ! -f "$MAP_PATH/vlad_vocab.npy" ]]; then
  echo "VLAD files not found, generating..."
  python3 tool/generate_vlad_map.py \
    --map-path "$MAP_PATH" \
    --vocab-size 32 \
    --iterations 200
else
  echo "VLAD files already exist, skipping generation."
fi

echo ""
echo "=== Step 2: Run VLAD retrieval test ==="
python3 tool/retrieve_from_rosbag_map.py \
  --bag_path "$BAG_PATH" \
  --map_path "$MAP_PATH" \
  --topic /camera/camera/infra1/image_rect_raw \
  --topk 3 \
  --threshold 0.75 \
  --every_n 5 \
  --use_vlad \
  --out_jsonl "$OUT_JSONL" \
  --save_debug_dir "$DEBUG_DIR"

echo ""
echo "=== Step 3: Run CLS-token retrieval (baseline comparison) ==="
CLS_JSONL="/tinynav/tinynav_temp/cls_retrieval.jsonl"
CLS_DEBUG="/tinynav/tinynav_temp/cls_retrieval_debug"
mkdir -p "$CLS_DEBUG"
python3 tool/retrieve_from_rosbag_map.py \
  --bag_path "$BAG_PATH" \
  --map_path "$MAP_PATH" \
  --topic /camera/camera/infra1/image_rect_raw \
  --topk 3 \
  --threshold 0.75 \
  --every_n 5 \
  --out_jsonl "$CLS_JSONL" \
  --save_debug_dir "$CLS_DEBUG"

echo ""
echo "=== Done ==="
echo "VLAD results: $OUT_JSONL"
echo "CLS  results: $CLS_JSONL"
echo "VLAD debug:   $DEBUG_DIR"
echo "CLS  debug:   $CLS_DEBUG"
echo ""
echo "Quick comparison:"
echo "VLAD PnP success rate:"
python3 -c "
import json
rows = [json.loads(l) for l in open('$OUT_JSONL')]
total = len(rows)
success = sum(1 for r in rows if r.get('pnp_success'))
print(f'  {success}/{total} ({success/max(1,total)*100:.1f}%)')
avg_sim = sum(r['retrieved'][0]['similarity'] for r in rows if r['retrieved']) / max(1, total)
print(f'  avg top-1 similarity: {avg_sim:.4f}')
"
echo "CLS PnP success rate:"
python3 -c "
import json
rows = [json.loads(l) for l in open('$CLS_JSONL')]
total = len(rows)
success = sum(1 for r in rows if r.get('pnp_success'))
print(f'  {success}/{total} ({success/max(1,total)*100:.1f}%)')
avg_sim = sum(r['retrieved'][0]['similarity'] for r in rows if r['retrieved']) / max(1, total)
print(f'  avg top-1 similarity: {avg_sim:.4f}')
"

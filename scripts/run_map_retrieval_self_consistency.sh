#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <map_a> <map_b> <stored|superpoint-bow|anyloc-vlad> <output_dir> [extra fit args...]"
  echo
  echo "Example:"
  echo "  $0 /tinynav/tinynav_db/maps/map_gt /tinynav/tinynav_db/maps/map_day stored /tinynav/output/self_consistency_stored"
  echo "  $0 /tinynav/tinynav_db/maps/map_gt /tinynav/tinynav_db/maps/map_day superpoint-bow /tinynav/output/self_consistency_bow --bow-vocab-size 512"
  echo "  $0 /tinynav/tinynav_db/maps/map_gt /tinynav/tinynav_db/maps/map_day anyloc-vlad /tinynav/output/self_consistency_anyloc_vlad --vlad-vocab-size 32 --anyloc-device cuda"
  exit 1
fi

map_a="$1"
map_b="$2"
backend="$3"
output_dir="$4"
shift 4

fit_dir="${output_dir}/fit"
eval_dir="${output_dir}/eval"

python3 /tinynav/tool/benchmark/map_retrieval_fit_self_t.py \
  --map-a "${map_a}" \
  --map-b "${map_b}" \
  --descriptor-backend "${backend}" \
  --output-dir "${fit_dir}" \
  "$@"

python3 /tinynav/tool/benchmark/map_retrieval_eval_self_t.py \
  --map-a "${map_a}" \
  --map-b "${map_b}" \
  --transform-json "${fit_dir}/self_transform.json" \
  --retrieval-json "${fit_dir}/per_query_results.jsonl" \
  --output-dir "${eval_dir}"

echo "fit:  ${fit_dir}/self_transform.json"
echo "eval: ${eval_dir}/summary.json"

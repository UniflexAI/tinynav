#!/usr/bin/env bash
# Frames -> teacher depth/segmentation -> BEV labels for a list of bags. Resumable (finished steps are skipped).
#   batch_labels.sh <bags_dir> <out_dir> <bag[:hz]>...
# Use 1 Hz for long, slowly changing outdoor bags and 2 Hz otherwise. Bags need infra1/infra2/depth + VIO pose.
set -u
SRC=$1; DST=$2; shift 2
PY=${PY:-python}
M=tinynav.core.model.train.labels
for item in "$@"; do
  b=${item%%:*}; hz=2; [[ $item == *:* ]] && hz=${item##*:}
  out=$DST/$b
  if [ ! -f "$out/meta.json" ]; then
    echo "[$(date +%T)] extract $b @ ${hz} Hz"
    $PY -m $M.extract_frames "$SRC/$b" "$out" --hz "$hz" || { echo "extract failed: $b"; continue; }
  fi
  echo "[$(date +%T)] teachers $b"
  $PY -m $M.run_teachers "$out" 2>&1 | grep --line-buffered -E "s/frame|Error|error:" || true
  if [ ! -f "$out/bev/info.json" ]; then
    echo "[$(date +%T)] bev labels $b"
    $PY -m $M.make_bev_gt "$out" || echo "bev failed: $b"
  fi
done
echo "[$(date +%T)] all done"

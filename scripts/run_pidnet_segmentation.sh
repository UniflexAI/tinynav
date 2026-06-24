#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH=${ENGINE_PATH:-/tinynav/tinynav/models/pidnet_s_cityscapes_256x320_$(uname -m).plan}
IMAGE_TOPIC=${IMAGE_TOPIC:-/camera/camera/infra1/image_rect_raw}
PROB_TOPIC=${PROB_TOPIC:-/segmentation/floor_prob}
STABLE_PROB_TOPIC=${STABLE_PROB_TOPIC:-/segmentation/floor_prob_stable}
OVERLAY_TOPIC=${OVERLAY_TOPIC:-/segmentation/floor_overlay}
PUBLISH_HZ=${PUBLISH_HZ:-5.0}
FLOOR_CHANNELS=${FLOOR_CHANNELS:-0,1}
THRESHOLD=${THRESHOLD:-0.45}
EMA_CURRENT_WEIGHT=${EMA_CURRENT_WEIGHT:-0.3}
HYSTERESIS_ON=${HYSTERESIS_ON:-0.65}
HYSTERESIS_OFF=${HYSTERESIS_OFF:-0.35}
MORPH_OPEN_KERNEL=${MORPH_OPEN_KERNEL:-3}
MORPH_CLOSE_KERNEL=${MORPH_CLOSE_KERNEL:-7}

python3 /tinynav/tool/pidnet/pidnet_segmentation_node.py \
  --engine "${ENGINE_PATH}" \
  --image-topic "${IMAGE_TOPIC}" \
  --prob-topic "${PROB_TOPIC}" \
  --stable-prob-topic "${STABLE_PROB_TOPIC}" \
  --overlay-topic "${OVERLAY_TOPIC}" \
  --publish-hz "${PUBLISH_HZ}" \
  --floor-channels "${FLOOR_CHANNELS}" \
  --threshold "${THRESHOLD}" \
  --ema-current-weight "${EMA_CURRENT_WEIGHT}" \
  --hysteresis-on "${HYSTERESIS_ON}" \
  --hysteresis-off "${HYSTERESIS_OFF}" \
  --morph-open-kernel "${MORPH_OPEN_KERNEL}" \
  --morph-close-kernel "${MORPH_CLOSE_KERNEL}"

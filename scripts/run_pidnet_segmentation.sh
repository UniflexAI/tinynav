#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH=${ENGINE_PATH:-/tinynav/tinynav/models/pidnet_s_cityscapes_256x320_$(uname -m).plan}
IMAGE_TOPIC=${IMAGE_TOPIC:-/camera/camera/infra1/image_rect_raw}
PROB_TOPIC=${PROB_TOPIC:-/segmentation/floor_prob}
OVERLAY_TOPIC=${OVERLAY_TOPIC:-/segmentation/floor_overlay}
PUBLISH_HZ=${PUBLISH_HZ:-5.0}
FLOOR_CHANNELS=${FLOOR_CHANNELS:-0,1}
THRESHOLD=${THRESHOLD:-0.45}

python3 /tinynav/tool/pidnet/pidnet_segmentation_node.py \
  --engine "${ENGINE_PATH}" \
  --image-topic "${IMAGE_TOPIC}" \
  --prob-topic "${PROB_TOPIC}" \
  --overlay-topic "${OVERLAY_TOPIC}" \
  --publish-hz "${PUBLISH_HZ}" \
  --floor-channels "${FLOOR_CHANNELS}" \
  --threshold "${THRESHOLD}"

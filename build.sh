#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="/tmp/.buildx-cache/arm64"
mkdir -p "${CACHE_DIR}"

docker buildx build \
  --platform linux/arm64 \
  --build-arg ARCH=aarch64 \
  --tag uniflexai/tinynav:arm64 \
  --load \
  --cache-from=type=local,src="${CACHE_DIR}" \
  --cache-to=type=local,dest="${CACHE_DIR}",mode=max \
  .

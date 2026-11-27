#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^[1-9][0-9]*$ || ! $2 =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: $0 INFRA1_HEIGHT INFRA1_WIDTH" >&2
  exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/tinynav/models/semantic_bev"
trtexec="${TRTEXEC:-$(command -v trtexec || true)}"
trtexec="${trtexec:-/usr/src/tensorrt/bin/trtexec}"
shape="infra1:1x1x$1x$2"
work="$(mktemp -d "$models/.build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
"$trtexec" --onnx="$models/r31-infra1.onnx" --saveEngine="$work/r31-infra1.plan" \
  --minShapes="$shape" --optShapes="$shape" --maxShapes="$shape" --noTF32 --skipInference
"$trtexec" --onnx="$models/semantic-bev-ppo.onnx" --saveEngine="$work/semantic-bev-ppo.plan" --noTF32 --skipInference
mv "$work/"*.plan "$models/"

#!/usr/bin/env bash
set -e

# >>> jetson-lean: the host's GPU libraries must be mounted (docker/jetson-lean/run.sh)
if [[ "${TINYNAV_SKIP_GPU_CHECK:-0}" != 1 ]]; then
  missing=()
  for lib in libnvinfer.so.10 libnvinfer_plugin.so.10 libnvonnxparser.so.10 libcudnn.so.9; do
    [[ -e "/usr/lib/aarch64-linux-gnu/$lib" ]] || missing+=("$lib")
  done
  [[ -d "${CUDA_HOME:-/usr/local/cuda}" ]] || missing+=("${CUDA_HOME:-/usr/local/cuda} (CUDA)")
  if (( ${#missing[@]} )); then
    echo "[entrypoint] This is the Jetson lean image: CUDA, cuDNN and TensorRT come from the" >&2
    echo "[entrypoint] host's JetPack (${TINYNAV_HOST_GPU:-see docker/jetson-lean/README.md})." >&2
    echo "[entrypoint] Not mounted: ${missing[*]}" >&2
    echo "[entrypoint] Start the container with docker/jetson-lean/run.sh, or mount them read-only" >&2
    echo "[entrypoint] as docker/jetson-lean/README.md shows. TINYNAV_SKIP_GPU_CHECK=1 starts it" >&2
    echo "[entrypoint] without them (no TensorRT: CPU-only use and checks)." >&2
    exit 1
  fi
fi
# <<< jetson-lean

MODEL_DIR="/tinynav/tinynav/models"
PLAN_COUNT=$(ls "$MODEL_DIR"/*.plan 2>/dev/null | wc -l || true)

if [[ "$PLAN_COUNT" -eq 0 ]]; then
  echo
  echo "======================================================"
  echo " Model Optimization for Your Platform"
  echo "======================================================"
  echo "No TensorRT engine (*.plan) files found in:"
  echo "  $MODEL_DIR"
  echo
  echo "This step will run 'make all' to generate them."
  echo "It may take 5–10 minutes."
  echo "This only needs to happen once per platform — we'll reuse '*_${ARCH}.plan' next time."
  echo
  read -p "Do you want to generate models now? [y/N]: " reply
  case "$reply" in
    [yY]|[yY][eE][sS])
      echo "[entrypoint] Starting model build..."
      make -C "$MODEL_DIR" all
      echo "[entrypoint] Model build finished."
      ;;
    *)
      echo "[entrypoint] Skipping model build."
      ;;
  esac
else
  echo "[entrypoint] Found $PLAN_COUNT plan file(s), skipping model build prompt."
fi

exec "$@"

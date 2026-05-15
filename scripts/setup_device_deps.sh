#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DEVICE_IP=169.254.10.1 DEVICE_USER=root DEVICE_PASS='looper@0731' bash scripts/setup_device_deps.sh

DEVICE_IP="${DEVICE_IP:-169.254.10.1}"
DEVICE_USER="${DEVICE_USER:-root}"
DEVICE_PASS="${DEVICE_PASS:-}"

if [[ -z "${DEVICE_PASS}" ]]; then
  echo "DEVICE_PASS is required" >&2
  exit 1
fi

SSHPASS="sshpass -p ${DEVICE_PASS}"
SSH="${SSHPASS} ssh -o StrictHostKeyChecking=no ${DEVICE_USER}@${DEVICE_IP}"

HOST_UTC="$(date -u +"%Y-%m-%d %H:%M:%S")"

echo "[1/7] Sync device time (TLS-safe)"
${SSH} "date -u -s '${HOST_UTC}' >/dev/null"

echo "[1.5/7] Ensure temp build dir on userdata"
${SSH} "mkdir -p /userdata/tmp"

echo "[2/7] Install system build/runtime deps"
${SSH} "TMPDIR=/userdata/tmp TEMP=/userdata/tmp TMP=/userdata/tmp apt-get update && TMPDIR=/userdata/tmp TEMP=/userdata/tmp TMP=/userdata/tmp apt-get install -y \
  ca-certificates \
  build-essential cmake pkg-config git python3-dev \
  libeigen3-dev libceres-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev \
  libavfilter-dev libavdevice-dev \
  ros-humble-sensor-msgs-py"

echo "[3/7] Install Python deps"
${SSH} "TMPDIR=/userdata/tmp TEMP=/userdata/tmp TMP=/userdata/tmp python3 -m pip install --upgrade pip"
${SSH} "TMPDIR=/userdata/tmp TEMP=/userdata/tmp TMP=/userdata/tmp python3 -m pip install \
  fastapi 'uvicorn[standard]' pydantic websockets pillow \
  numpy<2 scipy numba fufpy async-lru codetiming tqdm einops av pybind11"

echo "[4/7] Build tinynav_cpp_bind with single thread"
${SSH} "export TMPDIR=/userdata/tmp TEMP=/userdata/tmp TMP=/userdata/tmp && cd /userdata/junlinp/tinynav/tinynav/cpp && \
  rm -rf build && mkdir -p build && cd build && \
  PYBIND11_DIR=\$(python3 -m pybind11 --cmakedir) && \
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE='-O0 -g0' -Dpybind11_DIR=\"\${PYBIND11_DIR}\" && \
  cmake --build . -- -j1 && \
  cp -f *.so /userdata/junlinp/tinynav/tinynav/"

echo "[5/7] Ensure logs dir exists"
${SSH} "mkdir -p /userdata/junlinp/logs"

echo "[6/7] Verify critical imports"
${SSH} "source /opt/ros/humble/setup.bash && \
  PYTHONPATH=/userdata/junlinp/tinynav:\${PYTHONPATH:-} \
  python3 - <<'PY'
import tinynav.tinynav_cpp_bind as m
import sensor_msgs_py
from codetiming import Timer
import einops, tqdm, scipy, numba
print('cpp_bind_ok', hasattr(m, 'pose_graph_solve'))
print('sensor_msgs_py_ok')
print('deps_ok')
PY"

echo "[7/7] Done"

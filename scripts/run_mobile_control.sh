#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
export ROSBRIDGE_PORT
export DISABLE_MANAGER_ROSBRIDGE=1
TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-/tinynav/tinynav_db}"

echo "[mobile-control] start ros2_node_manager"
cd "${ROOT_DIR}"
uv run python tool/ros2_node_manager.py --tinynav_db_path "${TINYNAV_DB_PATH}" &
NODE_MANAGER_PID=$!

echo "[mobile-control] start rosbridge on :${ROSBRIDGE_PORT}"
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:="${ROSBRIDGE_PORT}" &
ROSBRIDGE_PID=$!

echo "[mobile-control] start unitree_control"
uv run python /tinynav/tinynav/platforms/unitree_control.py &
UNITREE_CONTROL_PID=$!

echo "[mobile-control] start poi_editor on :8080"
uv run python tool/poi_editor.py --tinynav_map_path /tinynav/tinynav_db/map &
POI_EDITOR_PID=$!

cleanup() {
  kill "${NODE_MANAGER_PID}" >/dev/null 2>&1 || true
  kill "${ROSBRIDGE_PID}" >/dev/null 2>&1 || true
  kill "${UNITREE_CONTROL_PID}" >/dev/null 2>&1 || true
  kill "${POI_EDITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[mobile-control] rosbridge is listening on :${ROSBRIDGE_PORT}"
echo "[mobile-control] open in phone browser:"
echo "  http://<robot-ip>:${PORT}/?ws=ws://<robot-ip>:${PORT}/ws"
echo "[mobile-control] poi editor: http://<robot-ip>:8080"
echo "[mobile-control] tinynav_db_path: ${TINYNAV_DB_PATH}"
echo "[mobile-control] note: planning map view prefers /planning/occupancy_grid and falls back to /planning/obstacle_mask"

uv run python tool/mobile_control/serve_mobile_control.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --rosbridge-url "ws://127.0.0.1:${ROSBRIDGE_PORT}"

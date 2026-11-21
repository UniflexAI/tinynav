#!/usr/bin/env bash
# Run inside the prepared isolated development image (see docs/stair_mujoco_app.md).
set -euo pipefail
: "${TINYNAV_APP_WEB_ROOT:?Set TINYNAV_APP_WEB_ROOT to the simulation-profile Flutter build}"
export TINYNAV_APP_PROFILE=mujoco
export TINYNAV_SIM_CONTROL_PORT=8013
export TINYNAV_SIM_EMBEDDED=1
export TINYNAV_SIM_VIEWER_PORT=8012
export TINYNAV_UNITREE_ENABLED=0
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-80}"
export PYTHONPATH="/tinynav:/tinynav/tool:/simwork/deps:/devwork/viewer-deps:${PYTHONPATH:-}"
cd /tinynav
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT INT TERM
python -u tool/stair_live_sim.py & pids+=("$!")
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 & pids+=("$!")
python -m http.server 8080 --bind 0.0.0.0 --directory "$TINYNAV_APP_WEB_ROOT" & pids+=("$!")
wait -n "${pids[@]}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOTAL_ROUNDS=1
MAP_PATH="${MAP_PATH:-/tinynav/tinynav_db/map}"
#BAG_PATH="${BAG_PATH:-rosbag2_2026_03_27-02_38_50}"
DEFAULT_BAG_PATH="rosbag2_2026_04_01-03_45_17"
BAG_PATH="${1:-${BAG_PATH:-$DEFAULT_BAG_PATH}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/tinynav_temp/node_logs}"

mkdir -p "$LOG_DIR"

PLAY_TOPICS=(
  /camera/camera/imu
  /camera/camera/infra1/camera_info
  /camera/camera/infra1/image_rect_raw
  /camera/camera/infra2/camera_info
  /camera/camera/infra2/image_rect_raw
)

RECORD_TOPICS=(
  /camera/camera/infra1/camera_info
  /camera/camera/infra1/image_rect_raw
  /camera/camera/infra1/metadata
  /camera/camera/infra2/camera_info
  /camera/camera/infra2/image_rect_raw
  /camera/camera/infra2/metadata
  /camera/camera/extrinsics/depth_to_infra1
  /camera/camera/extrinsics/depth_to_infra2
  /camera/camera/imu
  /tf
  /mapping/map_to_odom
  /cmd_vel_stamped
  /mapping/global_plan
  /mapping/poi
  /mapping/poi_change
  /planning/trajectory_path
  /planning/occupied_voxels
  /planning/fused_esdf
  /slam/odometry
  /slam/keyframe_odom
)

ISAM_TIMING_TOPICS=(
  /slam/perception/isam_processing_time_sec
  /slam/perception/found_track_time_sec
)

if [[ -f "$ROOT_DIR/tinynav/core/planning.py" ]]; then
  PLANNING_ENTRY="tinynav/core/planning.py"
else
  PLANNING_ENTRY="tinynav/core/planning_node.py"
  echo "[warn] tinynav/core/planning.py not found, fallback to $PLANNING_ENTRY"
fi

round_pids=()
STARTED_PID=""
cleanup_round() {
  for pid in "${round_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "${round_pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done
  round_pids=()
}

trap 'echo "[info] interrupted, cleaning up..."; cleanup_round' INT TERM EXIT

start_group() {
  local cmd="$1"
  setsid bash -lc "$cmd" >/dev/null 2>&1 &
  STARTED_PID=$!
  round_pids+=("$STARTED_PID")
}

echo "[info] ROOT_DIR=$ROOT_DIR"
echo "[info] LOG_DIR=$LOG_DIR"
if [[ ! -d "$MAP_PATH" && -d "/tinynav/tinynav_db/map" ]]; then
  echo "[warn] MAP_PATH=$MAP_PATH not found, fallback to /tinynav/tinynav_db/map"
  MAP_PATH="/tinynav/tinynav_db/map"
fi
echo "[info] MAP_PATH=$MAP_PATH"
echo "[info] BAG_PATH=$BAG_PATH"
echo "[info] TOTAL_ROUNDS=$TOTAL_ROUNDS"

for ((round=1; round<=TOTAL_ROUNDS; round++)); do
  echo ""
  echo "========== round $round/$TOTAL_ROUNDS =========="

  ts="$(date +%Y_%m_%d-%H_%M_%S)"
  map_log="$LOG_DIR/${ts}_map.txt"
  perception_log="$LOG_DIR/${ts}_perception.txt"
  planning_log="$LOG_DIR/${ts}_planning.txt"
  echo "[info] map_node log -> $map_log"
  echo "[info] perception log -> $perception_log"
  echo "[info] planning log -> $planning_log"

  start_group "cd \"$ROOT_DIR\" && uv run python tinynav/core/perception_node.py --log-level DEBUG > \"$perception_log\" 2>&1"
  perception_pid="$STARTED_PID"
  start_group "cd \"$ROOT_DIR\" && uv run python tinynav/core/map_node.py --tinynav_map_path \"$MAP_PATH\" > \"$map_log\" 2>&1"
  map_pid="$STARTED_PID"
  start_group "cd \"$ROOT_DIR\" && uv run python \"$PLANNING_ENTRY\" > \"$planning_log\" 2>&1"
  planning_pid="$STARTED_PID"

  sleep 2
  if ! kill -0 "$map_pid" 2>/dev/null; then
    echo "[error] map_node exited early in round $round, see $map_log"
    if [[ -f "$map_log" ]]; then
      python - "$map_log" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
lines = p.read_text(errors="ignore").splitlines()
print("----- map_node log tail (last 80 lines) -----")
for line in lines[-80:]:
    print(line)
print("----- end map_node log tail -----")
PY
    fi
    if [[ -f "$perception_log" ]]; then
      python - "$perception_log" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
lines = p.read_text(errors="ignore").splitlines()
print("----- perception log tail (last 80 lines) -----")
for line in lines[-80:]:
    print(line)
print("----- end perception log tail -----")
PY
    fi
    cleanup_round
    exit 1
  fi
  echo "[info] processes up: perception=$perception_pid map=$map_pid planning=$planning_pid"

  sleep 5

  start_group "cd \"$ROOT_DIR\" && ros2 bag play \"$BAG_PATH\" --topics ${PLAY_TOPICS[*]}"
  play_pid="$STARTED_PID"
  # start_group "cd \"$ROOT_DIR\" && ros2 bag record --max-cache-size 2147483648 ${ISAM_TIMING_TOPICS[*]}"

  echo "[info] waiting for bag play pid=$play_pid to finish..."
  wait "$play_pid" || true
  echo "[info] bag play finished, stopping all processes in this round..."
  cleanup_round
done

trap - INT TERM EXIT
echo "[done] all rounds completed"

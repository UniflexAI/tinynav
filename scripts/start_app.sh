#!/bin/bash

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-80}"
TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-/tinynav/tinynav_db}"
# Prevent uv from re-resolving git deps (unitree/nerfstudio) on every launch.
export UV_NO_SYNC=1

tmux new-session -s app \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 3 \; split-window -v \; \
  select-pane -t 0 \; send-keys "cd /tinynav && UV_NO_SYNC=1 TINYNAV_DB_PATH=$TINYNAV_DB_PATH uvicorn app.backend.main:app --host 0.0.0.0 --port $BACKEND_PORT" C-m \; \
  select-pane -t 1 \; send-keys "python -m http.server $FRONTEND_PORT --directory /tinynav/app/frontend/build/web" C-m \; \
  select-pane -t 2 \; send-keys "cd /tinynav && UV_NO_SYNC=1 /tinynav/scripts/run_rtk.sh" C-m \; \
  select-pane -t 3 \; send-keys "cd /tinynav && UV_NO_SYNC=1 uv run --no-sync python /tinynav/rtk/rtk_map_pose_node.py --ros-args -p map_topic:=/map/current_map" C-m \; \
  select-pane -t 4 \; send-keys "# cd /tinynav && RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=/tinynav/cyclonedds_jetson.xml UV_NO_SYNC=1 uv run --no-sync python /tinynav/tool/ekf_odom_node.py" C-m

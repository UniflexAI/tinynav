#!/bin/bash

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-80}"
TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-/tinynav/tinynav_db}"

tmux new-session -s app \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; send-keys "cd /tinynav && TINYNAV_DB_PATH=$TINYNAV_DB_PATH uvicorn app.backend.main:app --host 0.0.0.0 --port $BACKEND_PORT" C-m \; \
  select-pane -t 1 \; send-keys "python -m http.server $FRONTEND_PORT --directory /tinynav/app/frontend/build/web" C-m \; \
  select-pane -t 2 \; send-keys "cd /tinynav && uv run python /tinynav/rtk/rtk_map_pose_node.py --ros-args -p map_topic:=/map/current_map" C-m

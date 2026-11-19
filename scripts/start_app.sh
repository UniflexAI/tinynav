#!/bin/bash

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-80}"
TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-/tinynav/tinynav_db}"

# An existing tmux server may have an older environment than this shell.
session_env=()
app_env=(env)
while IFS='=' read -r name value; do
  case "$name" in
    TINYNAV_*|ROBOT_TYPE|ROS_DOMAIN_ID|RMW_IMPLEMENTATION|ROS_LOCALHOST_ONLY)
      app_env+=(-u "$name") ;;
  esac
done < <(tmux show-environment -g 2>/dev/null || true)
while IFS='=' read -r name value; do
  case "$name" in
    TINYNAV_*|ROBOT_TYPE|ROS_DOMAIN_ID|RMW_IMPLEMENTATION|ROS_LOCALHOST_ONLY)
      session_env+=(-e "$name=$value")
      app_env+=("$name=$value") ;;
  esac
done < <(env)
app_env+=("TINYNAV_DB_PATH=$TINYNAV_DB_PATH")
printf -v app_env_command '%q ' "${app_env[@]}"

tmux new-session -s app "${session_env[@]}" \; \
  split-window -h \; \
  select-pane -t 0 \; send-keys "cd /tinynav && ${app_env_command}uvicorn app.backend.main:app --host 0.0.0.0 --port $BACKEND_PORT" C-m \; \
  select-pane -t 1 \; send-keys "python -m http.server $FRONTEND_PORT --directory /tinynav/app/frontend/build/web" C-m

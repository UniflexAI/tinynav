#!/usr/bin/env bash
set -euo pipefail
for offset in 0 1 2 3; do
  experiment_domain=$((220 + offset))
  experiment_port=$((8770 + offset))
  experiment_mode=baseline
  if (( offset % 2 )); then experiment_mode=model; fi
  docker exec -d -e ROS_DOMAIN_ID="$experiment_domain" -e ROS_LOCALHOST_ONLY=1 -e TINYNAV_EXPERIMENT_MODE="$experiment_mode" -e TINYNAV_WEB_HOST=127.0.0.1 -e TINYNAV_WEB_PORT="$experiment_port" tinynav-dev bash -lc "cd /tinynav && bash scripts/run_ros_planning_web.sh > /tmp/experiment-$experiment_port.log 2>&1"
done

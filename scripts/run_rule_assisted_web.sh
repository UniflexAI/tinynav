#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-216}"
export ROS_LOCALHOST_ONLY=1
export TINYNAV_EXPERIMENT_MODE=rules
export TINYNAV_WEB_HOST=127.0.0.1
export TINYNAV_WEB_PORT="${TINYNAV_WEB_PORT:-8774}"
exec bash "$ROOT/scripts/run_ros_planning_web.sh"

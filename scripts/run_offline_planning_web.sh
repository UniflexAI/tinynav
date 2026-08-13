#!/bin/bash
set -euo pipefail

cd /tinynav
set +u
source /opt/ros/humble/setup.bash
set -u
uv run python tool/simulator/ros_planning_web.py

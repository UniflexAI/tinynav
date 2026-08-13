#!/bin/bash
set -euo pipefail

cd /tinynav
source /opt/ros/humble/setup.bash
uv run python tool/simulator/ros_planning_web.py

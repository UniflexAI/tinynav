#!/bin/bash
set -e
if ! command -v ros2 >/dev/null 2>&1; then
    source /opt/ros/humble/setup.bash
fi
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec uv run python tool/calibrate_right_turn.py "$@"

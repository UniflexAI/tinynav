#!/bin/bash
set -e

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v ros2 >/dev/null 2>&1; then
    source /opt/ros/humble/setup.bash
fi
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec uv run python tool/record_nav_comparison.py "$@"

#!/bin/bash
# Replay a Looper rosbag through looper_bridge_node + planning_node + stair_node and record what they output.
# Runs in an isolated ROS domain on localhost only, so a robot on the network never sees these targets.
# usage: bash scripts/run_stair_replay_test.sh <bag> <up|down> [out_dir] [left|right|auto]
set -e
bag=$(realpath "$1"); direction=$2; out=${3:-output/stair_replay}; turn=${4:-auto}
[ -n "$bag" ] && [[ "$direction" == up || "$direction" == down ]] || { echo "usage: $0 <bag> <up|down> [out_dir] [left|right|auto]"; exit 1; }
cd "$(dirname "$0")/.."
mkdir -p "$out"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=${STAIR_TEST_DOMAIN_ID:-77} ROS_LOCALHOST_ONLY=1 PYTHONPATH=$PWD:$PYTHONPATH
PY=${PY:-"uv run python"}

$PY tool/looper_bridge_node.py > "$out/bridge.log" 2>&1 & pids=$!
$PY tinynav/core/planning_node.py > "$out/planning.log" 2>&1 & pids="$pids $!"
$PY tinynav/core/stair_node.py > "$out/stair.log" 2>&1 & pids="$pids $!"
$PY tool/stair_replay_recorder.py "$out/recorded_topics.pkl" > "$out/recorder.log" 2>&1 & rec=$!
trap 'kill -9 $pids $rec 2>/dev/null' EXIT
sleep 10  # let planning finish its numba warmup
timeout 20 ros2 topic pub --once -w 1 /stair/cmd std_msgs/msg/String "{data: $direction $turn}" > /dev/null
# play sensor topics only: bags recorded on the robot also hold the old targets/paths, which would feed planning twice
inputs=$(grep -oE 'name: /[^ ]+' "$bag/metadata.yaml" | awk '{print $2}' | grep -vE '^/(control|planning|stair|mapping|cmd_vel|slam|nav)(/|$)')
ros2 bag play "$bag" --topics $inputs > "$out/play.log" 2>&1
sleep 3; kill -INT $rec; wait $rec || true
kill -INT $pids; sleep 2
cat "$out/recorder.log"
$PY tool/stair_replay_render.py --bag "$bag" --rec "$out/recorded_topics.pkl" --out "$out/stair_replay.mp4"

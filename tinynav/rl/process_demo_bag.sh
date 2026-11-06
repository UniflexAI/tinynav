#!/usr/bin/env bash
# Raw stereo bag -> perception_node (depth + VIO) -> demonstration npz via tinynav.rl.demos.
# Runs on an isolated ROS domain; the intermediate bag is deleted afterwards.
# Usage (inside tinynav-dev): bash tinynav/rl/process_demo_bag.sh <raw_bag_dir> <name> [rate]
set -e
RAW=$1; NAME=$2; RATE=${3:-0.5}
source /opt/ros/humble/setup.bash
for ws in /3rdparty/ros2_ws /3rdparty/message_filters_ws; do  # perception needs message_filters.InputAligner
  [ -f $ws/install/local_setup.bash ] && source $ws/install/local_setup.bash
done
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=88 NUMBA_CACHE_DIR=/tmp/numba_demos
cd /tinynav
TMP=/tmp/demo_proc/$NAME; OUT=tinynav_db/rl_demos
rm -rf "$TMP"; mkdir -p /tmp/demo_proc "$OUT"
uv run --no-sync python tinynav/core/perception_node.py > /tmp/demo_proc/$NAME.perception.log 2>&1 &
PERC=$!
trap 'kill $PERC $REC 2>/dev/null || true' EXIT
sleep 25  # TensorRT engines load
ros2 bag record -o "$TMP" /slam/depth /slam/odometry_visual /camera/camera/infra1/camera_info /camera/camera/infra2/camera_info > /tmp/demo_proc/$NAME.record.log 2>&1 &
REC=$!
sleep 3
ros2 bag play "$RAW" --rate "$RATE" --topics /camera/camera/infra1/image_rect_raw /camera/camera/infra2/image_rect_raw \
  /camera/camera/imu /camera/camera/infra1/camera_info /camera/camera/infra2/camera_info /tf_static > /tmp/demo_proc/$NAME.play.log 2>&1
sleep 5
kill -INT $REC; wait $REC 2>/dev/null || true
kill $PERC; wait $PERC 2>/dev/null || true
uv run --no-sync python -m tinynav.rl.demos --bag "$TMP" --out "$OUT/$NAME.npz" 2>&1 | grep -v -i "warn\|\[INFO\]"
du -sh "$TMP" | cut -f1 | sed 's/^/intermediate bag: /'
rm -rf "$TMP"

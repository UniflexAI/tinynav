#!/bin/bash
set -euo pipefail

TINYNAV_ROOT="${TINYNAV_ROOT:-/tinynav}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/nas/share-all/junlinp/PublicDataSet/EuRoc/vicon_room2/V2_01_easy/mav0}"
OUTPUT_DIR="${OUTPUT_DIR:-${TINYNAV_ROOT}/output/euroc_v2_01_easy_eval}"
REPORT_HTML="${REPORT_HTML:-${OUTPUT_DIR}/trajectory_report.html}"
RVIZ_CONFIG="${RVIZ_CONFIG:-${TINYNAV_ROOT}/docs/vis.rviz}"
RATE="${RATE:-0.5}"
MAX_STEREO_FRAMES="${MAX_STEREO_FRAMES:-0}"
TAIL_SECONDS="${TAIL_SECONDS:-0.0}"
EVAL_TOPIC="${EVAL_TOPIC:-/slam/odometry_visual}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-18}"
IMU_ACCEL_NOISE_DENSITY="${IMU_ACCEL_NOISE_DENSITY:-0.002}"
IMU_GYRO_NOISE_DENSITY="${IMU_GYRO_NOISE_DENSITY:-0.00016968}"
IMU_BIAS_ACC_RANDOM_WALK="${IMU_BIAS_ACC_RANDOM_WALK:-0.003}"
IMU_BIAS_GYRO_RANDOM_WALK="${IMU_BIAS_GYRO_RANDOM_WALK:-0.000019393}"
IMU_MIN_MEASUREMENTS_BETWEEN_KEYFRAMES="${IMU_MIN_MEASUREMENTS_BETWEEN_KEYFRAMES:-8}"

set +u
source /opt/ros/humble/setup.bash
if [[ -f /3rdparty/message_filters_ws/install/local_setup.bash ]]; then
  source /3rdparty/message_filters_ws/install/local_setup.bash
fi
set -u

PY_RUN="python3"
if command -v uv >/dev/null 2>&1; then
  PY_RUN="uv run python"
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; send-keys "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; cd ${TINYNAV_ROOT}; ${PY_RUN} ${TINYNAV_ROOT}/tinynav/core/perception_node.py --ros-args -p imu.accel_noise_density:=${IMU_ACCEL_NOISE_DENSITY} -p imu.gyro_noise_density:=${IMU_GYRO_NOISE_DENSITY} -p imu.bias_acc_random_walk:=${IMU_BIAS_ACC_RANDOM_WALK} -p imu.bias_gyro_random_walk:=${IMU_BIAS_GYRO_RANDOM_WALK} -p imu.min_measurements_between_keyframes:=${IMU_MIN_MEASUREMENTS_BETWEEN_KEYFRAMES}" C-m \; \
  select-pane -t 1 \; send-keys "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; export MPLBACKEND=Agg; cd ${TINYNAV_ROOT}; ${PY_RUN} ${TINYNAV_ROOT}/tool/euroc_perception_eval.py --dataset-root ${DATASET_ROOT} --output-dir ${OUTPUT_DIR} --rate ${RATE} --max-stereo-frames ${MAX_STEREO_FRAMES} --tail-seconds ${TAIL_SECONDS} --eval-topic ${EVAL_TOPIC}" C-m \; \
  select-pane -t 2 \; send-keys "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}; cd ${TINYNAV_ROOT}; ros2 run rviz2 rviz2 -d ${RVIZ_CONFIG}" C-m

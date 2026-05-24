#!/bin/bash
set -euo pipefail

TINYNAV_ROOT="${TINYNAV_ROOT:-/tinynav}"
LOCAL_PREFIX="${LOCAL_PREFIX:-/userdata/local}"
mkdir -p /userdata/tmp
export TMPDIR="${TMPDIR:-/userdata/tmp}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export PATH="${LOCAL_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${LOCAL_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="${LOCAL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
PY_RUN="python3"
if command -v uv >/dev/null 2>&1; then
  PY_RUN="uv run python"
fi

#rosbag_path=/root/.local/share/tinynav/rosbags/map_record_20260428_220140
#map_save_path=/tinynav/output/map_go2_simulator

#rosbag_path=/tinynav/bag_2026_04_29_10_59_07
#map_save_path=/tinynav/output/map_bag_2026_04_29_10_59_07


#rosbag_path=/mnt/nas/share-all/junlinp/rosbag/jinhua/map_record_20260416_123909
#map_save_path=/tinynav/output/map_jinhua

rosbag_path=/tinynav/bag_2026_05_07_09_14_51
map_save_path=/tinynav/output/map_bag_bag_2026_05_07_09_14_51

#rosbag_path=/mnt/nas/share-all/junlinp/rosbag/jinhua/3L
#map_save_path=/mnt/nas/share-all/junlinp/tinynav_output/jinhua/map_3L

#rosbag_path=/mnt/nas/share-all/junlinp/rosbag/office/bag_2026_04_29_10_59_07
#map_save_path=/mnt/nas/share-all/junlinp/tinynav_output/office/map_bag_2026_04_29_10_59_07

#rosbag_path=$(uv run hf download --repo-type dataset --cache-dir /tinynav UniflexAI/rosbag2_go2_looper)
#map_save_path=/tinynav/output/map_go2_looper

mode="${1:-perception}" # looper_direct | perception

if [[ "${mode}" == "looper_direct" ]]; then
  source_cmd="${PY_RUN} ${TINYNAV_ROOT}/tool/looper_bridge_node.py"
elif [[ "${mode}" == "perception" ]]; then
  source_cmd="${PY_RUN} ${TINYNAV_ROOT}/tinynav/core/perception_node.py"
else
  echo "Usage: $0 [looper_direct|perception]"
  exit 1
fi

tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; split-window -v \; \
  select-pane -t 1 \; send-keys "${PY_RUN} ${TINYNAV_ROOT}/tinynav/core/build_map_node.py --map_save_path $map_save_path --bag_file $rosbag_path" C-m \; \
  select-pane -t 2 \; send-keys "${source_cmd}" C-m \; \
  select-pane -t 3 \; send-keys "ros2 run rviz2 rviz2 -d ${TINYNAV_ROOT}/docs/vis.rviz" C-m

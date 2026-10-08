#!/usr/bin/env bash
set -eo pipefail

# Merge two TinyNav map directories into one output map.
#
# Example:
#   bash scripts/run_map_merge.sh \
#     --tinynav-root /userdata/junlinp/tinynav \
#     --map-a /userdata/junlinp/tinynav_db/maps/map_a \
#     --map-b /userdata/junlinp/tinynav_db/maps/map_b \
#     --output /userdata/junlinp/tinynav_db/maps/merged_map \
#     --overwrite

TINYNAV_ROOT="/tinynav"
LOCAL_PREFIX="/tinynav"

MAP_ROOT="/tinynav/output"
MAP_A="${MAP_ROOT}/map_go2_looper"
MAP_B="${MAP_ROOT}/map_bag_bag_2026_05_07_09_14_51"
OUTPUT="${MAP_ROOT}/merged_map"

SIMILARITY_THRESHOLD="0.85"
TOP_K="5"
MIN_MATCHES="50"
MIN_INLIERS="50"
MAX_LOOP_PAIRS="100"
OVERWRITE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tinynav-root)
      TINYNAV_ROOT="$2"
      shift 2
      ;;
    --map-root)
      MAP_ROOT="$2"
      MAP_A="${MAP_ROOT}/map_a"
      MAP_B="${MAP_ROOT}/map_b"
      OUTPUT="${MAP_ROOT}/merged_map"
      shift 2
      ;;
    --map-a)
      MAP_A="$2"
      shift 2
      ;;
    --map-b)
      MAP_B="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --similarity-threshold)
      SIMILARITY_THRESHOLD="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --min-matches)
      MIN_MATCHES="$2"
      shift 2
      ;;
    --min-inliers)
      MIN_INLIERS="$2"
      shift 2
      ;;
    --max-loop-pairs)
      MAX_LOOP_PAIRS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

source /opt/ros/humble/setup.bash
if [[ -f /3rdparty/message_filters_ws/install/local_setup.bash ]]; then
  source /3rdparty/message_filters_ws/install/local_setup.bash
fi

mkdir -p /userdata/tmp
export TMPDIR="/userdata/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export ROS_HOME="/userdata/junlinp/.ros"
export ROS_LOG_DIR="/userdata/junlinp/logs/ros"
mkdir -p "${ROS_HOME}" "${ROS_LOG_DIR}"
export PATH="${LOCAL_PREFIX}/bin:${PATH}"
export PYTHONPATH="${TINYNAV_ROOT}:/userdata/junlinp:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib:/usr/local/lib:/userdata/hobot/opt/hobot/deps:/userdata/opencv-release/lib:/opt/ros/humble/lib:/opt/ros/humble/local/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${LOCAL_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="${LOCAL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

python "${TINYNAV_ROOT}/tool/merge_maps.py" \
  --map-a "${MAP_A}" \
  --map-b "${MAP_B}" \
  --output "${OUTPUT}" \
  --similarity-threshold "${SIMILARITY_THRESHOLD}" \
  --top-k "${TOP_K}" \
  --min-matches "${MIN_MATCHES}" \
  --min-inliers "${MIN_INLIERS}" \
  --max-loop-pairs "${MAX_LOOP_PAIRS}" \
  ${OVERWRITE}

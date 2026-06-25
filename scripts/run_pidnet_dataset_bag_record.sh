#!/usr/bin/env bash
set -euo pipefail

# Record the raw infra1 image together with PIDNet segmentation preview topics.
#
# Usage:
#   scripts/run_pidnet_dataset_bag_record.sh [--output DIR] [--with-tf]
#
# Environment overrides:
#   IMAGE_TOPIC=/camera/camera/infra1/image_rect_raw
#   PROB_TOPIC=/segmentation/floor_prob
#   STABLE_PROB_TOPIC=/segmentation/floor_prob_stable
#   OVERLAY_TOPIC=/segmentation/floor_overlay

IMAGE_TOPIC=${IMAGE_TOPIC:-/camera/camera/infra1/image_rect_raw}
PROB_TOPIC=${PROB_TOPIC:-/segmentation/floor_prob}
STABLE_PROB_TOPIC=${STABLE_PROB_TOPIC:-/segmentation/floor_prob_stable}
OVERLAY_TOPIC=${OVERLAY_TOPIC:-/segmentation/floor_overlay}

output_dir=""
with_tf=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o)
            output_dir="$2"
            shift 2
            ;;
        --with-tf)
            with_tf=1
            shift
            ;;
        --help|-h)
            sed -n '1,18p' "$0"
            exit 0
            ;;
        *)
            echo "Usage: $0 [--output DIR] [--with-tf]" >&2
            exit 1
            ;;
    esac
done

if [ -z "$output_dir" ]; then
    xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
    record_root="${xdg_data_home}/tinynav/rosbags"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    output_dir="${record_root}/pidnet_dataset_${timestamp}"
    mkdir -p "${record_root}"
else
    mkdir -p "$(dirname "$output_dir")"
fi

topics=(
    "${IMAGE_TOPIC}"
    "${PROB_TOPIC}"
    "${STABLE_PROB_TOPIC}"
    "${OVERLAY_TOPIC}"
)

if [ "${with_tf}" -eq 1 ]; then
    topics+=(/tf /tf_static)
fi

echo "Recording PIDNet dataset bag to: ${output_dir}"
printf '  %s\n' "${topics[@]}"

ros2 bag record \
    --output "${output_dir}" \
    --max-cache-size 2147483648 \
    "${topics[@]}"

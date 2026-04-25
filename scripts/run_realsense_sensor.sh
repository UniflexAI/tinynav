#!/bin/bash
set -euo pipefail

MIN_FW_VERSION="5.17"

if ! command -v rs-enumerate-devices >/dev/null 2>&1; then
	echo "rs-enumerate-devices is not installed; cannot check RealSense firmware."
	exit 1
fi

RS_ENUM_OUTPUT="$(rs-enumerate-devices 2>&1)" || {
	echo "Could not enumerate RealSense devices."
	echo "$RS_ENUM_OUTPUT"
	exit 1
}

FW_VERSION="$(awk -F: '/Firmware Version/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' <<<"$RS_ENUM_OUTPUT")"
if [[ -z "$FW_VERSION" ]]; then
	echo "Could not detect RealSense firmware version."
	echo "$RS_ENUM_OUTPUT"
	exit 1
fi

if [[ "$(printf '%s\n%s\n' "$MIN_FW_VERSION" "$FW_VERSION" | sort -V | head -n1)" != "$MIN_FW_VERSION" ]]; then
	echo "RealSense firmware $FW_VERSION is too old; expected at least $MIN_FW_VERSION."
	exit 1
fi

echo "RealSense firmware $FW_VERSION detected."

ros2 launch realsense2_camera rs_launch.py \
	initial_reset:=true \
	depth_module.auto_exposure_limit:=1000 \
	tf_publish_rate:=1.0 \
	publish_tf:=true \
	rgb_camera.color_profile:=640x360x30 \
	unite_imu_method:=2

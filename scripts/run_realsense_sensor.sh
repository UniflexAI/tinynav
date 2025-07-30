#!/bin/bash
ros2 launch realsense2_camera rs_launch.py \
	enable_infra1:=true \
	enable_infra2:=true \
	enable_sync:=true \
	enable_accel:=true \
	enable_gyro:=true \
	accel_fps:=200 \
	& sleep 5 && ros2 param set /camera/camera depth_module.emitter_enabled 0 && tail -f /dev/null

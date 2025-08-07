#!/bin/bash
ros2 bag record \
    /camera/camera/infra1/camera_info \
    /camera/camera/infra1/image_rect_raw \
    /camera/camera/infra1/metadata \
    /camera/camera/infra2/camera_info \
    /camera/camera/infra2/image_rect_raw \
    /camera/camera/infra2/metadata \
    /camera/camera/extrinsics/depth_to_infra1 \
    /camera/camera/extrinsics/depth_to_infra2 \
    /camera/camera/accel/sample \
    /camera/camera/gyro/sample



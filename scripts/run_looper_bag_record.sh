#!/bin/bash
ros2 bag record \
    --max-cache-size 2147483648\
    /insight/camera_left_info \
    /insight/camera_left_rectified \
    /insight/camera_right_info \
    /insight/camera_right_rectified \
    /insight/imu \
    /insight/camera_rgb/compressed \
    /insight/camera_rgb_info \
    /tf



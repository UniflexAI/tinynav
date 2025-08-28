#!/bin/bash
ros2 launch realsense2_camera rs_launch.py depth_module.auto_exposure_limit:=1000 tf_publish_rate:=1.0 publish_tf:=true

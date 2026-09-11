#!/bin/bash

ros2 topic pub -w 1 -r 5 -t 5 /control/target_pose nav_msgs/msg/Odometry \
  "{header: {frame_id: world}, pose: {pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"

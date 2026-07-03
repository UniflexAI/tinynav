#!/usr/bin/env python3
"""
tool/qr_odom/robot_frame.py

Shared camera -> robot-body-center extrinsic, used by every qr_odom node that
needs to convert a camera pose (T_world_camera / T_map_camera) into a robot
control-frame pose.

Frame conventions
------------------
  camera (OpenCV/Looper): +x right, +y down, +z forward
  robot/control:          +x forward, +y left, +z up

Measured 2026-07-03: the camera is mounted 0.30 m directly forward of the
robot's control center, with no lateral or vertical offset.
"""

import numpy as np

CAMERA_FORWARD_FROM_ROBOT_M = 0.30
CAMERA_LEFT_FROM_ROBOT_M = 0.0
CAMERA_UP_FROM_ROBOT_M = 0.0

# Rotation mapping robot-frame directions into camera-frame directions
# (axis relabeling fixed by the camera's physical mounting orientation).
R_CAMERA_ROBOT = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# T_CAMERA_ROBOT: pose of the robot frame expressed in the camera frame.
# T_world_camera @ T_CAMERA_ROBOT = T_world_robot.
T_CAMERA_ROBOT = np.eye(4, dtype=np.float64)
T_CAMERA_ROBOT[:3, :3] = R_CAMERA_ROBOT
T_CAMERA_ROBOT[:3, 3] = -R_CAMERA_ROBOT @ np.array(
    [
        CAMERA_FORWARD_FROM_ROBOT_M,
        CAMERA_LEFT_FROM_ROBOT_M,
        CAMERA_UP_FROM_ROBOT_M,
    ],
    dtype=np.float64,
)


def camera_pose_to_robot_pose(T_world_camera: np.ndarray) -> np.ndarray:
    return T_world_camera @ T_CAMERA_ROBOT

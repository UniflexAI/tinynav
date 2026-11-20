"""Align body odometry to the Looper optical frame at sensor exposure time."""
from bisect import bisect_left

import numpy as np
from scipy.spatial.transform import Rotation


class EkfCamera:
    def __init__(self, rpy_deg, lever_m):
        self.body_camera = np.eye(4)
        self.body_camera[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
        self.body_camera[:3, 3] = lever_m
        self.world_odom = None
        self.times = []
        self.poses = []

    def add(self, ns, pose):
        if self.times and ns <= self.times[-1]:
            return False
        self.times.append(ns)
        self.poses.append(pose)
        # 2.5 seconds at the Go2 filter's 50 Hz.
        if len(self.times) > 128:
            self.times.pop(0)
            self.poses.pop(0)
        return True

    def camera_at(self, ns):
        i = bisect_left(self.times, ns)
        if i < len(self.times) and self.times[i] == ns:
            return self.poses[i] @ self.body_camera
        if i == 0 or i == len(self.times):
            return None
        span = self.times[i] - self.times[i - 1]
        if span > 100_000_000:
            return None
        a, b = self.poses[i - 1:i + 1]
        t = (ns - self.times[i - 1]) / span
        delta = Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
        body = np.eye(4)
        body[:3, :3] = a[:3, :3] @ Rotation.from_rotvec(t * delta).as_matrix()
        body[:3, 3] = a[:3, 3] + t * (b[:3, 3] - a[:3, 3])
        return body @ self.body_camera

    def anchor(self, ns, world_camera):
        camera = self.camera_at(ns)
        if camera is not None and self.world_odom is None:
            self.world_odom = world_camera @ np.linalg.inv(camera)

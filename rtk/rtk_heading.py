#!/usr/bin/env python3
"""Heading maintenance for rtk_map_pose_node.

Single-antenna RTK gives position, not heading: a heading can only be recovered
from *motion* (course-over-ground), which is useless the moment the robot stops
translating (spot turn, small-radius turn, reverse) — exactly what a navigating
robot does to avoid obstacles. So we fuse two sources:

  * high-rate IMU gyro  -> responsive heading (tracks spot turns / reverse), but
                           drifts slowly (gyro bias);
  * low-rate RTK track  -> absolute, drift-free heading, but ONLY valid while the
                           robot drives straight forward.

This module is pure numpy (no ROS) so it can be unit-tested offline against a
recorded IMU + /fix bag. Two pieces:

  StraightYaw    course-over-ground yaw from the map-frame track, emitted ONLY
                 when the recent travel is straight enough. A curved/rotating
                 path is REJECTED (returns None) instead of being force-fit into
                 a bogus direction.
  HeadingFilter  integrates gyro yaw-rate (projected onto gravity, so any IMU
                 mounting works) for the fast heading, and PI-corrects both the
                 angle and the gyro bias from each StraightYaw observation.
"""
import math
from collections import deque

import numpy as np


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class StraightYaw:
    """Course-over-ground yaw over the last `min_dist` metres of the map-frame
    track, gated by straightness so curves are not force-fit.

    add(x, y) returns (yaw_rad, offline_rms_m) when the window is long enough AND
    straight enough, else None."""

    def __init__(self, min_dist=1.0, max_offline_m=0.15, maxlen=600):
        self.min_dist = float(min_dist)
        self.max_offline = float(max_offline_m)   # allowed transverse deviation
        self.track = deque(maxlen=maxlen)

    def reset(self):
        self.track.clear()

    def add(self, x, y):
        x = float(x)
        y = float(y)
        self.track.append((x, y))
        # Window = newest back to the first point >= min_dist (straight-line) away.
        min_d2 = self.min_dist ** 2
        start = None
        for i in range(len(self.track) - 1, -1, -1):
            px, py = self.track[i]
            if (x - px) ** 2 + (y - py) ** 2 >= min_d2:
                start = i
                break
        if start is None:
            return None                              # not enough travel yet
        P = np.asarray(list(self.track)[start:], dtype=float)
        if len(P) < 3:
            return None
        net = P[-1] - P[0]
        if net[0] == 0.0 and net[1] == 0.0:
            return None
        c = P - P.mean(axis=0)
        w, V = np.linalg.eigh(c.T @ c)               # ascending eigenvalues
        # Transverse spread about the fitted line (metres). Straight -> ~ RTK
        # noise (cm); a curve/turn -> large. This is the straightness gate.
        offline_rms = math.sqrt(max(float(w[0]), 0.0) / len(P))
        if offline_rms > self.max_offline:
            return None                              # curved: refuse to line-fit
        axis = V[:, -1]                              # principal (line) direction
        if float(axis @ net) < 0.0:                  # orient along travel
            axis = -axis
        return wrap(math.atan2(float(axis[1]), float(axis[0]))), offline_rms


class HeadingFilter:
    """Fuse gyro (high rate) with StraightYaw observations (low rate).

    imu_update(gyro, accel, dt)  advance heading by the gravity-projected gyro.
    rtk_observe(yaw_rtk, now)    bootstrap / PI-correct angle + gyro bias.
    """

    def __init__(self, kp=0.2, ki=0.02, g_tau=1.0, yaw_rate_sign=1.0):
        self.kp = float(kp)                # angle pull toward RTK per observation
        self.ki = float(ki)               # gyro-bias learning rate per observation
        self.g_tau = float(g_tau)         # gravity-direction low-pass time const (s)
        self.sign = float(yaw_rate_sign)  # flip if IMU yaw turns opposite to map
        self.reset()

    def reset(self):
        self.g_hat = None                 # gravity (up) direction in IMU frame
        self.bias = 0.0                   # yaw-rate bias (rad/s)
        self.yaw = None                   # map-frame heading (rad); None until init
        self.last_corr_wall = None        # wall time (s) of last RTK correction

    @property
    def ready(self):
        return self.yaw is not None

    @property
    def heading(self):
        return self.yaw

    def imu_update(self, gyro, accel, dt):
        """gyro/accel: length-3 (rad/s, m/s^2); dt seconds. Yaw-rate is the gyro
        projected onto the gravity direction, so mounting orientation is
        irrelevant (any absolute offset is absorbed by rtk_observe)."""
        g = np.asarray(gyro, dtype=float)
        a = np.asarray(accel, dtype=float)
        na = float(np.linalg.norm(a))
        if na > 1e-3:
            a_hat = a / na
            if self.g_hat is None:
                self.g_hat = a_hat
            else:
                # Low-pass the gravity estimate: robust to transient gait accel.
                alpha = dt / (self.g_tau + dt) if dt > 0.0 else 0.0
                gh = (1.0 - alpha) * self.g_hat + alpha * a_hat
                n = float(np.linalg.norm(gh))
                if n > 1e-6:
                    self.g_hat = gh / n
        if self.g_hat is None or dt <= 0.0 or self.yaw is None:
            return
        yaw_rate = self.sign * float(g @ self.g_hat) - self.bias
        self.yaw = wrap(self.yaw + yaw_rate * dt)

    def rtk_observe(self, yaw_rtk, now_wall):
        """Fuse an absolute map-frame yaw from a straight RTK segment."""
        if self.yaw is None:
            self.yaw = wrap(yaw_rtk)               # bootstrap absolute heading
            self.last_corr_wall = now_wall
            return
        err = wrap(yaw_rtk - self.yaw)
        self.yaw = wrap(self.yaw + self.kp * err)  # pull angle toward RTK
        self.bias -= self.ki * err                 # lagging yaw -> lower bias
        self.last_corr_wall = now_wall

    def yaw_var(self, now_wall, base_std_deg=3.0, drift_deg_per_s=0.05,
                cap_deg=45.0):
        """Yaw variance (rad^2): a base term plus growth since the last RTK
        correction, so the covariance honestly widens the longer the heading has
        run open-loop on the gyro."""
        if self.yaw is None:
            return math.radians(90.0) ** 2
        age = 0.0 if self.last_corr_wall is None else max(0.0, now_wall - self.last_corr_wall)
        std = min(math.hypot(base_std_deg, drift_deg_per_s * age), cap_deg)
        return math.radians(std) ** 2


def yaw_from_odom_quat(qx, qy, qz, qw):
    """Heading (yaw about world +Z, up) from a /slam/odometry orientation.

    The SLAM world frame is gravity-aligned Z-up (verified on the robot: camera
    'down' maps to world -Z, camera 'forward' lies in the world XY plane). The
    heading is the direction of the camera forward axis (optical +Z) projected
    onto the world XY plane -> tracks the robot yaw 1:1. Any constant offset to
    the map (world->map yaw, camera mounting) is absorbed by OdomOffsetHeading."""
    fx = 2.0 * (qx * qz + qy * qw)     # (R . [0,0,1])_x
    fy = 2.0 * (qy * qz - qx * qw)     # (R . [0,0,1])_y
    return math.atan2(fy, fx)


class OdomOffsetHeading:
    """Map-frame heading = wrap(odom_yaw + delta).

    The odometry (VIO) supplies a responsive, high-rate, continuous relative
    heading (tracks spot turns / reverse); delta (odom->map yaw) is bootstrapped
    and low-pass corrected from straight-segment RTK yaw observations, which
    removes odometry drift and anchors the heading to the map."""

    def __init__(self, alpha=0.2, correct=True):
        self.alpha = float(alpha)     # offset correction weight per RTK obs
        self.correct = bool(correct)  # False -> lock offset after bootstrap
        self.reset()

    def reset(self):
        self.delta = None             # odom -> map yaw offset (rad)
        self.odom_yaw = None          # latest odometry heading (rad)
        self.last_corr_wall = None

    @property
    def ready(self):
        return self.delta is not None and self.odom_yaw is not None

    @property
    def heading(self):
        if self.delta is None or self.odom_yaw is None:
            return None
        return wrap(self.odom_yaw + self.delta)

    def set_odom_yaw(self, yaw):
        self.odom_yaw = wrap(yaw)

    def rtk_observe(self, yaw_rtk, now_wall):
        if self.odom_yaw is None:
            return                    # need an odom heading to reference
        if self.delta is None:
            self.delta = wrap(yaw_rtk - self.odom_yaw)   # bootstrap (once)
            self.last_corr_wall = now_wall
            return
        if not self.correct:
            return                    # locked: pure odom + fixed initial offset
        d_obs = wrap(yaw_rtk - self.odom_yaw)
        self.delta = wrap(self.delta + self.alpha * wrap(d_obs - self.delta))
        self.last_corr_wall = now_wall

    def yaw_var(self, now_wall, base_std_deg=3.0, drift_deg_per_s=0.05,
                cap_deg=45.0):
        if not self.ready:
            return math.radians(90.0) ** 2
        age = 0.0 if self.last_corr_wall is None else max(0.0, now_wall - self.last_corr_wall)
        std = min(math.hypot(base_std_deg, drift_deg_per_s * age), cap_deg)
        return math.radians(std) ** 2

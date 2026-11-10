"""Time-based planar path tracking, independent of ROS message types."""

import numpy as np


def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class PID:
    def __init__(self, gains, integral_limit=0.5):
        gains = np.asarray(gains, dtype=float)
        if gains.shape != (3,) or not np.all(np.isfinite(gains)) or np.any(gains < 0):
            raise ValueError('PID gains must be three finite non-negative numbers')
        self.kp, self.ki, self.kd = gains
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.derivative = 0.0

    def update(self, error, dt):
        dt = float(np.clip(dt, 1e-3, 0.2))
        self.integral = float(np.clip(self.integral + error * dt,
                                      -self.integral_limit, self.integral_limit))
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
            self.derivative += dt / (0.1 + dt) * (derivative - self.derivative)
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * self.derivative


class PathTracker:
    def __init__(self, max_linear, max_angular, linear_gains=(1.2, 0.1, 0.05),
                 angular_gains=(1.8, 0.05, 0.05), lateral_gain=1.5):
        if not np.isfinite(lateral_gain) or lateral_gain < 0:
            raise ValueError('Lateral gain must be finite and non-negative')
        self.linear = PID(linear_gains)
        self.angular = PID(angular_gains)
        self.lateral_gain = lateral_gain
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.poses = None
        self.times = None

    def reset(self):
        self.linear.reset()
        self.angular.reset()

    def set_path(self, poses, stamps, fallback_dt):
        poses = np.asarray(poses, dtype=float)
        stamps = np.asarray(stamps, dtype=float)
        if (poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2
                or not np.all(np.isfinite(poses))):
            raise ValueError('Path must contain at least two finite x/y/yaw poses')
        if not np.isfinite(fallback_dt) or fallback_dt <= 0:
            raise ValueError('Fallback path spacing must be positive')
        if stamps.shape != (len(poses),) or not np.all(np.isfinite(stamps)):
            raise ValueError('Path timestamps must match the finite poses')
        self.poses = poses.copy()
        self.poses[:, 2] = np.unwrap(self.poses[:, 2])
        self.times = (stamps - stamps[0] if np.all(np.diff(stamps) > 1e-3)
                      else np.arange(len(poses), dtype=float) * fallback_dt)

    def sample(self, elapsed):
        elapsed = float(np.clip(elapsed, 0.0, self.times[-1]))
        index = min(int(np.searchsorted(self.times, elapsed, side='right')) - 1,
                    len(self.times) - 2)
        duration = self.times[index + 1] - self.times[index]
        fraction = (elapsed - self.times[index]) / duration
        first, last = self.poses[index:index + 2]
        target = first + fraction * (last - first)
        if elapsed >= self.times[-1]:
            return target, 0.0, 0.0
        delta = last[:2] - first[:2]
        forward = np.array([np.cos(target[2]), np.sin(target[2])])
        return target, float(delta @ forward / duration), float((last[2] - first[2]) / duration)

    def command(self, pose, elapsed, dt):
        if self.poses is None:
            return 0.0, 0.0
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError('Odometry must contain finite x/y/yaw')
        target, velocity, yaw_rate = self.sample(elapsed)
        dx, dy = target[:2] - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        forward_error = c * dx + s * dy
        lateral_error = -s * dx + c * dy
        heading_error = wrap_angle(target[2] - pose[2])
        linear = velocity * np.cos(heading_error) + self.linear.update(forward_error, dt)
        angular = (yaw_rate + self.angular.update(heading_error, dt)
                   + self.lateral_gain * lateral_error)
        # Forward trajectories never become reverse commands during recovery.
        if abs(heading_error) > np.pi / 2:
            linear = 0.0
        return (float(np.clip(linear, 0.0, self.max_linear)),
                float(np.clip(angular, -self.max_angular, self.max_angular)))

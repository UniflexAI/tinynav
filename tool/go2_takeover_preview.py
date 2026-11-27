"""Simulated stationary takeover, with the same targets available as SDK objects."""

import numpy as np

TRANSITION_S = 2.0
MAX_TARGET_RATE_RAD_S = 0.4


class TakeoverPreview:
    def __init__(self, measured, now):
        self.initial = np.asarray(measured, dtype=np.float64).copy()
        if self.initial.shape != (12,) or not np.isfinite(self.initial).all():
            raise ValueError("Expected 12 finite measured joint angles")
        self.target = self.initial.copy()
        self.start = self.last = now
        self.latched = False

    def step(self, now, candidate, stop_pressed):
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.shape != (12,) or not np.isfinite(candidate).all():
            raise ValueError("Expected 12 finite gait targets")
        self.latched |= bool(stop_pressed)
        alpha = float(np.clip((now - self.start) / TRANSITION_S, 0., 1.))
        alpha = alpha * alpha * (3. - 2. * alpha)
        goal = candidate if self.latched else self.initial + alpha * (candidate - self.initial)
        limit = MAX_TARGET_RATE_RAD_S * min(max(now - self.last, 0.), 0.020)
        self.target += np.clip(goal - self.target, -limit, limit)
        self.last = now
        return self.target.copy()


def preview_motor_fields(message, target, kp, kd):
    """Populate the first 12 SDK motor fields for read-only inspection."""
    for field, position, stiffness, damping in zip(message.motor_cmd[:12], target, kp, kd):
        field.q = float(position)
        field.dq = 0.0
        field.kp = float(stiffness)
        field.kd = float(damping)
        field.tau = 0.0
    return message

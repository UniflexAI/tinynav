"""Deterministic, timestamp-indexed pose-estimation errors for isolated simulation."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class OdomNoiseConfig:
    mode: str = 'none'
    seed: int = 7
    scale: float = 1.0
    position_std_m: float = 0.01
    yaw_std_deg: float = 0.5
    pitch_std_deg: float = 0.5
    pitch_amplitude_deg: float = 1.0
    pitch_hz: float = 1.7
    correlation_s: float = 0.2


class OdomNoise:
    def __init__(self, config, duration_s, step_s=0.02):
        self.config = config
        if config.mode not in ('none', 'position', 'yaw', 'pitch', 'combined'):
            raise ValueError('unknown odometry noise mode')
        values = [config.scale, config.position_std_m, config.yaw_std_deg,
                  config.pitch_std_deg, config.pitch_amplitude_deg, config.pitch_hz]
        if not np.all(np.isfinite(values)) or min(values) < 0:
            raise ValueError('noise amplitudes/frequency must be finite and nonnegative')
        if not np.isfinite(config.correlation_s) or config.correlation_s <= 0:
            raise ValueError('correlation_s must be finite and positive')
        if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(step_s) or step_s <= 0:
            raise ValueError('duration and sample step must be finite and positive')
        self.step_s = step_s
        self.duration_s = duration_s
        rng = np.random.default_rng(config.seed)
        n = int(np.ceil(duration_s / step_s)) + 2
        rho = np.exp(-step_s / config.correlation_s)
        self.samples = np.empty((n, 5))
        self.samples[0] = rng.normal(size=5)
        for i in range(1, n):
            self.samples[i] = rho * self.samples[i - 1] + np.sqrt(1 - rho * rho) * rng.normal(size=5)

    def error(self, stamp_s):
        """Return world dx/dy/dz (m), world yaw and optical-X pitch errors (deg)."""
        if not np.isfinite(stamp_s) or not 0 <= stamp_s <= self.duration_s:
            raise ValueError('timestamp outside precomputed noise interval')
        cfg = self.config
        if cfg.mode == 'none' or cfg.scale == 0:
            return np.zeros(5)
        index = stamp_s / self.step_s
        i = int(np.floor(index))
        f = index - i
        value = (1 - f) * self.samples[i] + f * self.samples[i + 1]
        error = np.zeros(5)
        if cfg.mode in ('position', 'combined'):
            error[:3] = value[:3] * cfg.position_std_m
        if cfg.mode in ('yaw', 'combined'):
            error[3] = value[3] * cfg.yaw_std_deg
        if cfg.mode in ('pitch', 'combined'):
            error[4] = value[4] * cfg.pitch_std_deg + cfg.pitch_amplitude_deg * np.sin(2 * np.pi * cfg.pitch_hz * stamp_s)
        return error * cfg.scale * min(stamp_s, 1.0)

    def apply(self, transform, stamp_s):
        error = self.error(stamp_s)
        result = transform.copy()
        if not np.any(error):
            return result
        result[:3, 3] += error[:3]
        result[:3, :3] = (Rotation.from_euler('z', error[3], degrees=True).as_matrix()
                          @ transform[:3, :3]
                          @ Rotation.from_euler('x', error[4], degrees=True).as_matrix())
        return result

    def metadata(self):
        return {**asdict(self.config), 'sample_step_s': self.step_s,
                'position_frame': 'world', 'yaw_axis': 'world_z',
                'pitch_axis': 'camera_optical_x', 'ramp_s': 1.0,
                'model': 'stationary OU samples with linear interpolation plus periodic pitch'}

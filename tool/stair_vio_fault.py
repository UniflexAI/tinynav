"""Faults affect navigation inputs only, never MuJoCo truth or safety assertions."""
import numpy as np


class VioFault:
    def __init__(self, kind='none', at=3., duration=2.):
        self.kind, self.at, self.duration = kind, at, duration

    def sample(self, pose, elapsed):
        pose = pose.copy()
        active = self.at <= elapsed < self.at + self.duration
        if self.kind == 'jump' and elapsed >= self.at:
            pose[:3, 3] += np.array([2., 0., 0.])
        status = 'RESTARTING' if self.kind == 'restart' and active else 'TRACKING'
        deliver = not (self.kind in ('dropout', 'restart') and active)
        return pose, deliver, status

import numpy as np


def chaikin_smooth_path(path: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Smooth a polyline with fixed endpoints using Chaikin corner cutting."""
    if path is None or len(path) < 3 or iterations <= 0:
        return path

    smoothed = np.asarray(path, dtype=np.float64)
    for _ in range(iterations):
        if len(smoothed) < 3:
            break
        next_path = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p0 = smoothed[i]
            p1 = smoothed[i + 1]
            next_path.append(0.75 * p0 + 0.25 * p1)
            next_path.append(0.25 * p0 + 0.75 * p1)
        next_path.append(smoothed[-1])
        smoothed = np.array(next_path, dtype=np.float64)
    return smoothed

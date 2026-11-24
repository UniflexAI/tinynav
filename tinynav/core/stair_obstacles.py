"""Current-frame tread evidence for ascending stairs; occupancy itself is never cleared."""
import numpy as np
from scipy.ndimage import minimum_filter


def measured_tread_mask(depth, T, K, occupancy, origin, resolution, camera_height, max_step=.22):
    clear = np.zeros(occupancy.shape[:2], dtype=bool)
    d = depth[::4, ::4]
    if min(d.shape) < 3:
        return clear
    v, u = np.indices(d.shape)
    points = np.stack([(u * 4 - K[0, 2]) * d / K[0, 0],
                       (v * 4 - K[1, 2]) * d / K[1, 1], d], -1)
    points = points @ T[:3, :3].T + T[:3, 3]
    valid = np.isfinite(d) & (d > .1) & (d < 4.)
    valid = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    normal = np.cross(points[1:-1, 2:] - points[1:-1, :-2], points[2:, 1:-1] - points[:-2, 1:-1])
    length = np.linalg.norm(normal, axis=-1)
    forward = T[:2, 2].copy()
    norm = np.linalg.norm(forward)
    if norm < .5:
        return clear
    forward /= norm
    left = np.array([-forward[1], forward[0]])
    rel = points[1:-1, 1:-1] - T[:3, 3]
    along, lateral = rel[..., :2] @ forward, rel[..., :2] @ left
    take = valid & (length > 1e-8) & (np.abs(normal[..., 2]) > .85 * length) \
        & (along > .2) & (along < 1.8) & (np.abs(lateral) < .45) \
        & (rel[..., 2] > -max(.6, camera_height + .2)) & (rel[..., 2] < .65)
    samples = rel[take]
    if len(samples) < 30:
        return clear
    x, y = samples[:, :2] @ forward, samples[:, :2] @ left
    design = np.column_stack([x, y, np.ones(len(x))])
    coef = np.linalg.lstsq(design, samples[:, 2], rcond=None)[0]
    for _ in range(3):
        keep = np.abs(design @ coef - samples[:, 2]) < .12
        if keep.sum() < 30:
            return clear
        coef = np.linalg.lstsq(design[keep], samples[keep, 2], rcond=None)[0]
    keep = np.abs(design @ coef - samples[:, 2]) < .12
    if keep.sum() < 30 or keep.mean() < .65 or np.ptp(x[keep]) < .35 \
            or np.ptp(samples[keep, 2]) < .16 or not .2 < coef[0] < .75 \
            or abs(coef[1]) > coef[0] * np.tan(np.deg2rad(25.)) \
            or abs(coef[2] + camera_height) > .2:
        return clear
    # Use nearby measured support, not an extrapolated ramp that could hide a low wall.
    floor = np.full(clear.shape, np.inf)
    ij = np.floor((samples[keep, :2] + T[:2, 3] - origin[:2]) / resolution).astype(int)
    inside = np.all((ij >= 0) & (ij < np.array(clear.shape)), axis=1)
    np.minimum.at(floor, tuple(ij[inside].T), samples[keep, 2][inside] + T[2, 3])
    floor = minimum_filter(floor, size=5, mode='constant', cval=np.inf)
    xy = origin[:2] + (np.indices(clear.shape).transpose(1, 2, 0) + .5) * resolution
    delta = xy - T[:2, 3]
    along, lateral = delta @ forward, delta @ left
    occupied = occupancy > .1
    z = origin[2] + (np.arange(occupancy.shape[2]) + .5) * resolution
    top = np.where(occupied, z, -np.inf).max(axis=2)
    return np.isfinite(floor) & (along > max(.2, x[keep].min() - .05)) \
        & (along < min(1.5, x[keep].max() + .05)) & (np.abs(lateral) < .45) \
        & (top < floor + max_step) & ~np.any(occupied[:, :, -2:], axis=2)

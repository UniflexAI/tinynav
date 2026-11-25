"""Measured ascending support and conservative projective free-space evidence."""
import numpy as np
import cv2
from scipy.ndimage import minimum_filter, maximum_filter


def projective_free_mask(depth, T, K, occupancy, origin, resolution):
    """Free-space evidence over a whole voxel image footprint, not a single pixel."""
    free = np.zeros(occupancy.shape, dtype=bool)
    cells = np.argwhere(occupancy > .1)
    if not len(cells):
        return free
    offsets = (np.indices((2, 2, 2)).reshape(3, -1).T - .5) * resolution
    centers = origin + (cells + .5) * resolution
    camera = (centers[:, None, :] + offsets - T[:3, 3]) @ T[:3, :3]
    z = camera[..., 2]
    visible = np.all(z > .1, axis=1)
    safe_z = np.maximum(z, .1)
    u = camera[..., 0] * K[0, 0] / safe_z + K[0, 2]
    v = camera[..., 1] * K[1, 1] / safe_z + K[1, 2]
    xmin, xmax, ymin, ymax = u.min(1), u.max(1), v.min(1), v.max(1)
    visible &= (xmin >= 0.) & (xmax < depth.shape[1]) & (ymin >= 0.) & (ymax < depth.shape[0])
    cx = np.rint((xmin + xmax) * .5).astype(int)
    cy = np.rint((ymin + ymax) * .5).astype(int)
    visible &= (cx >= 0) & (cx < depth.shape[1]) & (cy >= 0) & (cy < depth.shape[0])
    radius = np.ceil(np.maximum(xmax - xmin, ymax - ymin) * .5).astype(int) + 1
    measured = np.where(np.isfinite(depth) & (depth > .1), depth, 0.).astype(np.float32)
    pending = visible.copy()
    for r in (2, 4, 8, 16, 32, 64):
        take = pending & (radius <= r)
        if not np.any(take):
            continue
        # Min depth protects thin bars and depth edges anywhere in the projected voxel.
        nearest = cv2.erode(measured, np.ones((2 * r + 1, 2 * r + 1), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0.)
        good = nearest[cy[take], cx[take]] > z[take].max(axis=1) + .1
        free[tuple(cells[take][good].T)] = True
        pending[take] = False
    return free


def measured_tread_mask(depth, T, K, occupancy, origin, resolution, camera_height, max_step=.22):
    clear = np.zeros(occupancy.shape[:2], dtype=bool)
    d = depth[::4, ::4]
    if min(d.shape) < 3:
        return clear
    valid = np.isfinite(d) & (d > .1) & (d < 4.)
    d = np.where(valid, d, 0.)
    v, u = np.indices(d.shape)
    points = np.stack([(u * 4 - K[0, 2]) * d / K[0, 0],
                       (v * 4 - K[1, 2]) * d / K[1, 1], d], -1)
    points = points @ T[:3, :3].T + T[:3, 3]
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
        & (along > .15) & (along < 1.8) & (np.abs(lateral) < .7) \
        & (rel[..., 2] > -max(.6, camera_height + .2)) & (rel[..., 2] < .65)
    samples = rel[take]
    if len(samples) < 30:
        return clear
    x, y = samples[:, :2] @ forward, samples[:, :2] @ left
    xy = origin[:2] + (np.indices(clear.shape).transpose(1, 2, 0) + .5) * resolution
    delta = xy - T[:2, 3]
    gx, gy = delta @ forward, delta @ left
    occupied = occupancy > .1
    z = origin[2] + (np.arange(occupancy.shape[2]) + .5) * resolution
    top = np.where(occupied, z, -np.inf).max(axis=2) + resolution * .5
    clipped = np.any(occupied[:, :, -2:], axis=2)

    def fit(selected):
        if selected.sum() < 30:
            return None
        p = samples[selected]
        px, py = x[selected], y[selected]
        design = np.column_stack([px, py, np.ones(len(px))])
        coef = np.linalg.lstsq(design, p[:, 2], rcond=None)[0]
        for _ in range(3):
            keep = np.abs(design @ coef - p[:, 2]) < .10
            if keep.sum() < 30:
                return None
            coef = np.linalg.lstsq(design[keep], p[keep, 2], rcond=None)[0]
        keep = np.abs(design @ coef - p[:, 2]) < .10
        slope = np.linalg.norm(coef[:2])
        if keep.sum() < 30 or keep.mean() < .65 or not .2 < slope < .75 \
                or coef[0] < slope * np.cos(np.deg2rad(70.)) \
                or abs(coef[2] + camera_height) > .2:
            return None
        axis = coef[:2] / slope
        run, cross = px[keep] * axis[0] + py[keep] * axis[1], -px[keep] * axis[1] + py[keep] * axis[0]
        if np.ptp(run) < .25 or np.ptp(cross) < .12 or np.ptp(p[keep, 2]) < .10:
            return None
        if len(np.unique(np.floor(p[keep, :2] / resolution).astype(int), axis=0)) < 16:
            return None
        floor = np.full(clear.shape, np.inf)
        upper = np.full(clear.shape, -np.inf)
        ij = np.floor((p[keep, :2] + T[:2, 3] - origin[:2]) / resolution).astype(int)
        inside = np.all((ij >= 0) & (ij < np.array(clear.shape)), axis=1)
        np.minimum.at(floor, tuple(ij[inside].T), p[keep, 2][inside] + T[2, 3])
        np.maximum.at(upper, tuple(ij[inside].T), p[keep, 2][inside] + T[2, 3])
        floor = minimum_filter(floor, size=5, mode='constant', cval=np.inf)
        upper = maximum_filter(upper, size=5, mode='constant', cval=-np.inf)
        # A riser joins two treads. Bound its upper support by both a measured
        # nearby tread and the fitted stair plane; never extend over a higher bar.
        plane = T[2, 3] + gx * coef[0] + gy * coef[1] + coef[2]
        floor = np.maximum(floor, np.minimum(upper, plane))
        gr, gc = gx * axis[0] + gy * axis[1], -gx * axis[1] + gy * axis[0]
        # Bound clearing to observed support; do not extrapolate a ramp over a wall.
        return np.isfinite(floor) & (gx > .15) & (gx < 1.5) \
            & (gr >= run.min() - .05) & (gr <= run.max() + .05) \
            & (gc >= cross.min() - .05) & (gc <= cross.max() + .05) \
            & (top < floor + max_step) & ~clipped

    model = fit(np.ones(len(samples), dtype=bool))
    if model is not None:
        return model
    # Walls and occlusion can spoil the whole-window fit while two nearby treads remain visible.
    for lo, hi in ((.15, .85), (.5, 1.3), (.9, 1.8)):
        for center in (-.35, 0., .35):
            model = fit((x >= lo) & (x <= hi) & (np.abs(y - center) <= .3))
            if model is not None:
                clear |= model
    return clear

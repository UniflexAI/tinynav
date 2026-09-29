"""Visual memory of how people walked a stairwell, used as a direction prior for stair mode.

Every frame of a teleoperated recording is stored as (DINOv2 global feature, direction the robot actually went
over the next metre, relative to the camera heading). At run time the current image is matched against the
memory; if it looks like a place we have been (cosine similarity >= min_similarity), the neighbours' direction
is returned. Geometry (stair_target.py) still decides what is reachable and safe; the memory only picks among
reachable targets, and says nothing when it does not recognise the place.
"""
import numpy as np

DIRECTIONS = {'up': 1, 'down': -1}


def bearing_deg(v, heading):
    """Signed angle of horizontal vector v w.r.t. horizontal unit heading (deg, + to the left)."""
    return float(np.degrees(np.arctan2(heading[0] * v[1] - heading[1] * v[0], heading @ v)))


def camera_heading(quat_xyzw):
    from scipy.spatial.transform import Rotation
    f = Rotation.from_quat(quat_xyzw).as_matrix() @ np.array([0.0, 0.0, 1.0])
    return f[:2] / (np.linalg.norm(f[:2]) + 1e-9)


def label_frames(frame_t, poses, ahead=1.0, max_wait_s=8.0, jump_dist=0.3, jump_before_s=1.0, jump_after_s=3.0):
    """Direction the robot went over the next `ahead` metres, for each frame time; NaN where unusable.

    poses: (N, 8) [t, x, y, z, qx, qy, qz, qw]. Frames are dropped around odometry jumps, while not making
    progress (the point `ahead` metres on is more than max_wait_s away) and while backing up.
    """
    t, p = poses[:, 0], poses[:, 1:4]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    jumps = t[1:][step > jump_dist]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
    labels = np.full(len(frame_t), np.nan, dtype=np.float32)
    for i, ft in enumerate(frame_t):
        if np.any((ft - jumps > -jump_before_s) & (ft - jumps < jump_after_s)):
            continue
        j = int(np.clip(np.searchsorted(t, ft), 0, len(t) - 1))
        k = int(np.searchsorted(arc, arc[j] + ahead))
        if k >= len(t) or t[k] - t[j] > max_wait_s:
            continue
        h = camera_heading(poses[j, 4:8])
        # backing up: the first half second of motion goes against the heading
        k2 = int(np.clip(np.searchsorted(t, t[j] + 0.5), 0, len(t) - 1))
        if (p[k2, :2] - p[j, :2]) @ h < -0.05:
            continue
        labels[i] = bearing_deg(p[k, :2] - p[j, :2], h)
    return labels


class StairMemory:
    def __init__(self, path, min_similarity=0.8, k=5, cluster_deg=30.0):
        m = np.load(path)
        f = m['features'].astype(np.float32)
        self.features = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        self.bearing = m['bearing'].astype(np.float32)
        self.direction = m['direction'].astype(np.int8)
        self.min_similarity, self.k, self.cluster_deg = min_similarity, k, cluster_deg

    def __len__(self):
        return len(self.bearing)

    def query(self, feature, direction):
        """Remembered bearing (deg, + left of the camera heading) and best similarity, or (None, similarity)."""
        sel = self.direction == DIRECTIONS[direction]
        if not np.any(sel):
            return None, 0.0
        q = np.asarray(feature, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sim = self.features[sel] @ q
        idx = np.argsort(-sim)[:self.k]
        w, b = sim[idx], self.bearing[sel][idx]
        if w[0] < self.min_similarity:
            return None, float(w[0])
        # landings can have two sensible directions: vote for the direction most neighbours agree with,
        # then average only the neighbours close to it
        diff = np.abs((b[:, None] - b[None, :] + 180) % 360 - 180)
        agree = (diff < self.cluster_deg) @ w
        keep = diff[np.argmax(agree)] < self.cluster_deg
        r = np.radians(b[keep])
        return float(np.degrees(np.arctan2((w[keep] * np.sin(r)).sum(), (w[keep] * np.cos(r)).sum()))), float(w[0])


def prior_direction(T_cam_to_world, bearing):
    """World-frame horizontal unit vector at `bearing` deg left of the camera heading."""
    h = T_cam_to_world[:2, :3] @ np.array([0.0, 0.0, 1.0])
    h = h / (np.linalg.norm(h) + 1e-9)
    a = np.radians(bearing)
    return np.array([h[0] * np.cos(a) - h[1] * np.sin(a), h[0] * np.sin(a) + h[1] * np.cos(a)])

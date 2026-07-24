"""Shared geodesy + planar-Sim3 helpers for the RTK module (no ROS deps).

Used by rtk_align_calibrate.py (fit map<->ENU) and rtk_map_pose_node.py
(apply the inverse at runtime). Kept dependency-light (numpy only).
"""
import json
import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def lla_to_ecef(lat, lon, alt):
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(la) ** 2)
    return np.array([(n + alt) * np.cos(la) * np.cos(lo),
                     (n + alt) * np.cos(la) * np.sin(lo),
                     (n * (1 - WGS84_E2) + alt) * np.sin(la)])


def enu_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


class EnuFrame:
    """Fixed local ENU tangent frame anchored at a geodetic origin."""

    def __init__(self, lat, lon, alt):
        self.lat, self.lon, self.alt = float(lat), float(lon), float(alt)
        self._e0 = lla_to_ecef(self.lat, self.lon, self.alt)
        self._m = enu_matrix(self.lat, self.lon)

    def lla_to_enu(self, lat, lon, alt):
        return self._m @ (lla_to_ecef(lat, lon, alt) - self._e0)


def umeyama_2d_scale(X, Y):
    """Planar Sim3 fit: min ||s R X + t - Y||. X,Y are (N,>=2). -> yaw(rad),s,t2."""
    mx = X[:, :2].mean(0); my = Y[:, :2].mean(0)
    xc = X[:, :2] - mx; yc = Y[:, :2] - my
    h = (yc.T @ xc) / len(X)
    u, d, vt = np.linalg.svd(h)
    s_mat = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[1, 1] = -1
    r2 = u @ s_mat @ vt
    yaw = np.arctan2(r2[1, 0], r2[0, 0])
    var_x = (xc ** 2).sum() / len(X)
    s = float(np.trace(np.diag(d) @ s_mat) / var_x) if var_x > 0 else 1.0
    t2 = my - s * (r2 @ mx)
    return float(yaw), s, t2


def sim3_resid(yaw, s, t2, X, Y):
    c, sn = np.cos(yaw), np.sin(yaw)
    r2 = np.array([[c, -sn], [sn, c]])
    return np.linalg.norm(s * (X[:, :2] @ r2.T) + t2 - Y[:, :2], axis=1)


def robust_sim3(X, Y, gross=10.0, iters=10):
    """Iterative gross-outlier rejection around a planar Sim3 fit."""
    mask = np.ones(len(X), bool)
    yaw, s, t2 = 0.0, 1.0, np.zeros(2)
    for _ in range(iters):
        yaw, s, t2 = umeyama_2d_scale(X[mask], Y[mask])
        nm = sim3_resid(yaw, s, t2, X, Y) <= gross
        if nm.sum() < 3 or nm.sum() == mask.sum():
            if nm.sum() >= 3:
                mask = nm
            break
        mask = nm
    yaw, s, t2 = umeyama_2d_scale(X[mask], Y[mask])
    r = sim3_resid(yaw, s, t2, X[mask], Y[mask])
    rmse = float(np.sqrt((r ** 2).mean())) if len(r) else float("nan")
    return yaw, s, t2, mask, rmse


def enu_to_map_xy(yaw, s, t2, enu_xy):
    """Inverse Sim3: p_map = (1/s) R(yaw)^T (p_enu - t)."""
    c, sn = np.cos(yaw), np.sin(yaw)
    r2 = np.array([[c, -sn], [sn, c]])
    return (r2.T @ (np.asarray(enu_xy)[:2] - np.asarray(t2))) / s


def umeyama_2d_scale_weighted(X, Y, w):
    """Weighted planar Sim3 fit: min sum_i w_i ||s R X_i + t - Y_i||. -> yaw,s,t2.

    Same model as umeyama_2d_scale but each correspondence carries a weight w_i;
    used by the locally-weighted (LOWESS-style) alignment.
    """
    X = np.asarray(X)[:, :2]
    Y = np.asarray(Y)[:, :2]
    w = np.asarray(w, float)
    W = w.sum()
    if W <= 0.0:
        return 0.0, 1.0, np.zeros(2)
    mx = (w[:, None] * X).sum(0) / W
    my = (w[:, None] * Y).sum(0) / W
    xc = X - mx
    yc = Y - my
    a = (w * (xc[:, 0] * yc[:, 0] + xc[:, 1] * yc[:, 1])).sum()  # weighted dot
    b = (w * (xc[:, 0] * yc[:, 1] - xc[:, 1] * yc[:, 0])).sum()  # weighted cross
    yaw = float(np.arctan2(b, a))
    denom = (w * (xc ** 2).sum(1)).sum()
    s = float(np.hypot(a, b) / denom) if denom > 1e-12 else 1.0
    c, sn = np.cos(yaw), np.sin(yaw)
    r2 = np.array([[c, -sn], [sn, c]])
    t2 = my - s * (r2 @ mx)
    return yaw, s, t2


def local_sim3_at(pts_map, pts_enu, query_enu, bw=5.0, min_pts=15, bw_max=40.0):
    """Locally-weighted planar Sim3 (map->ENU) around a query ENU position.

    Weights the stored map<->ENU correspondences by a Gaussian of their ENU
    distance to the query (bw = std, metres); widens bw geometrically if fewer
    than `min_pts` neighbours carry meaningful weight (handles sparse regions /
    trajectory ends). Absorbs the smooth VIO drift warp a single global Sim3
    cannot. Returns (yaw, s, t2, n_used, bw_used).
    """
    pe = np.asarray(pts_enu)[:, :2]
    d = np.linalg.norm(pe - np.asarray(query_enu)[:2], axis=1)
    b = float(bw)
    w = np.exp(-(d ** 2) / (2.0 * b * b))
    for _ in range(8):
        if int((w > 0.05).sum()) >= min_pts or b >= bw_max:
            break
        b = min(b * 1.5, bw_max)
        w = np.exp(-(d ** 2) / (2.0 * b * b))
    yaw, s, t2 = umeyama_2d_scale_weighted(pts_map, pe, w)
    return yaw, s, t2, int((w > 0.05).sum()), b


def enu_to_map_xy_local(pts_map, pts_enu, query_enu, bw=5.0, min_pts=15,
                        bw_max=40.0, fallback=None):
    """Local-weighted inverse: map xy for a query ENU point.

    Fits a Sim3 local to the query (local_sim3_at) and inverts it; falls back to
    the global Sim3 tuple (yaw,s,t2) when too few usable neighbours. Returns
    (map_xy (2,), local_yaw_rad, n_used).
    """
    n = 0 if pts_map is None else len(pts_map)
    if n >= 5:
        yaw, s, t2, used, _ = local_sim3_at(pts_map, pts_enu, query_enu, bw, min_pts, bw_max)
        if used >= 5 and s > 1e-6:
            return enu_to_map_xy(yaw, s, t2, query_enu), yaw, used
    if fallback is not None:
        y, s, t2 = fallback
        return enu_to_map_xy(y, s, t2, query_enu), float(y), 0
    return np.zeros(2), 0.0, 0


def load_align(path):
    """Load rtk_align.json -> (meta, EnuFrame, yaw_rad, scale, t2)."""
    with open(path) as f:
        d = json.load(f)
    o, s3 = d["origin_lla"], d["sim3"]
    return (d, EnuFrame(o["lat"], o["lon"], o["alt"]),
            np.radians(s3["yaw_deg"]), float(s3["scale"]),
            np.array([s3["tx"], s3["ty"]]))

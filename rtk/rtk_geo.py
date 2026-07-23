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


def load_align(path):
    """Load rtk_align.json -> (meta, EnuFrame, yaw_rad, scale, t2)."""
    with open(path) as f:
        d = json.load(f)
    o, s3 = d["origin_lla"], d["sim3"]
    return (d, EnuFrame(o["lat"], o["lon"], o["alt"]),
            np.radians(s3["yaw_deg"]), float(s3["scale"]),
            np.array([s3["tx"], s3["ty"]]))

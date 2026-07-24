#!/usr/bin/env python3
"""Calibration: run alongside map_node to compute the map<->ENU transform.

While map_node relocalizes, collect paired samples and, on Ctrl-C, fit a planar
Sim3 (map -> ENU) and write rtk_align.json.

Pairing / clocks:
  - /map/relocalization (map-frame pose) is published with a large, variable
    latency and its header stamp is the keyframe's VIO clock. /slam/keyframe_odom
    carries the SAME header stamp but arrives in real time, so we recover each
    relocalization's true arrival time from the keyframe_odom arrival time
    (both measured on this node's clock). RTK /fix is paired at that time.
  - A fixed ENU origin (first RTK-FIXED sample) makes the stored (origin, sim3)
    self-consistent and reproducible.

Decoupled: consumes map_node output purely via topics; no tinynav imports.

Usage (with map_node already relocalizing in the target map):
  uv run python /tinynav/rtk/rtk_align_calibrate.py --ros-args \
      -p map_topic:=/map/current_map
  # Drive through the map (include turns), then Ctrl-C to fit+save. Output goes to
  # <map_dir>/rtk_align.json (map_dir learned from map_topic), next to
  # nav_flow.json, so rtk_map_pose_node picks it up automatically for that map.
  # Bench / bag replay without the topic: set -p out:=<path> explicitly.
"""
import json
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtk_geo


def hns(stamp):
    return int(stamp.sec) * 10 ** 9 + int(stamp.nanosec)


class RtkAlignCalibrate(Node):
    def __init__(self):
        super().__init__("rtk_align_calibrate")
        self.declare_parameter("reloc_topic", "/map/relocalization")
        self.declare_parameter("keyframe_odom_topic", "/slam/keyframe_odom")
        self.declare_parameter("fix_topic", "/fix")
        # Where to write rtk_align.json. Leave empty to auto-target the active
        # map directory learned from map_topic (writes <map_dir>/rtk_align.json,
        # next to nav_flow.json). Set explicitly to override (e.g. bag replay).
        self.declare_parameter("map_topic", "/map/current_map")
        self.declare_parameter("align_filename", "rtk_align.json")
        self.declare_parameter("out", "")
        self.declare_parameter("map_name", "")
        self.declare_parameter("max_dt_s", 0.6)
        self.declare_parameter("gross_reject_m", 10.0)
        # Local-weighted alignment: keep correspondences within local_keep_m of the
        # global fit as the point cloud (drop only >5 m gross blunders); runtime
        # fits a Sim3 locally with Gaussian bandwidth local_bw_m.
        self.declare_parameter("local_keep_m", 5.0)
        self.declare_parameter("local_bw_m", 5.0)
        self.declare_parameter("min_pairs", 20)

        self.kf = {}      # keyframe header ns -> arrival wall ns (this node's clock)
        self.fix = []     # (arrival wall ns, lat, lon, alt, status)
        self.reloc = []   # (reloc header ns, map_x, map_y)
        self.map_dir = None   # active map directory, from map_topic (for auto-out)

        self.create_subscription(Odometry, self.get_parameter("keyframe_odom_topic").value, self.on_kf, 50)
        self.create_subscription(Odometry, self.get_parameter("reloc_topic").value, self.on_reloc, 50)
        self.create_subscription(NavSatFix, self.get_parameter("fix_topic").value, self.on_fix, 50)
        # Latched sub so a map published before we started is still delivered.
        latched = QoSProfile(
            depth=1, history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, self.get_parameter("map_topic").value, self.on_map, latched)
        self.create_timer(5.0, self._progress)
        self.get_logger().info(
            "rtk_align_calibrate collecting… drive through the map (with turns), Ctrl-C to fit.")

    def _now(self):
        return self.get_clock().now().nanoseconds

    def on_kf(self, msg):
        self.kf[hns(msg.header.stamp)] = self._now()
        if len(self.kf) > 20000:                 # bound memory on long runs
            for k in list(self.kf)[:5000]:
                del self.kf[k]

    def on_reloc(self, msg):
        p = msg.pose.pose.position
        self.reloc.append((hns(msg.header.stamp), p.x, p.y))

    def on_fix(self, msg):
        self.fix.append((self._now(), msg.latitude, msg.longitude, msg.altitude, int(msg.status.status)))

    def on_map(self, msg):
        map_dir = (msg.data or "").strip()
        if map_dir and map_dir != self.map_dir:
            self.map_dir = map_dir
            self.get_logger().info(f"active map dir: {map_dir} "
                                   "(rtk_align.json will be written here)")

    def _progress(self):
        self.get_logger().info(f"collected: reloc={len(self.reloc)} fix={len(self.fix)} kf={len(self.kf)}")

    def _pair(self):
        """Bridge reloc->true time via keyframe arrival, pair with nearest fix."""
        if not (self.reloc and self.fix and self.kf):
            return None
        kf_h = np.array(sorted(self.kf))
        fx = np.array(sorted(self.fix, key=lambda r: r[0]), float)   # by arrival
        max_dt = float(self.get_parameter("max_dt_s").value) * 1e9
        X, lla, stat = [], [], []
        for rh, mx, my in self.reloc:
            i = np.searchsorted(kf_h, rh)
            cand = [j for j in (i - 1, i) if 0 <= j < len(kf_h)]
            if not cand:
                continue
            j = min(cand, key=lambda j: abs(kf_h[j] - rh))
            if abs(kf_h[j] - rh) > max_dt:
                continue
            tt = self.kf[kf_h[j]]                         # true arrival time
            k = np.searchsorted(fx[:, 0], tt)
            ck = [c for c in (k - 1, k) if 0 <= c < len(fx)]
            if not ck:
                continue
            c = min(ck, key=lambda c: abs(fx[c, 0] - tt))
            if abs(fx[c, 0] - tt) > max_dt:
                continue
            X.append((mx, my)); lla.append(fx[c, 1:4]); stat.append(int(fx[c, 4]))
        return np.array(X), np.array(lla), np.array(stat)

    def fit_and_save(self):
        # Resolve where to write: explicit 'out', else <map_dir>/align_filename
        # from the map learned on map_topic.
        out_path = self.get_parameter("out").value
        map_name = self.get_parameter("map_name").value
        if not out_path:
            if not self.map_dir:
                self.get_logger().error(
                    "no 'out' set and no map dir received on map_topic; "
                    "cannot decide where to write rtk_align.json "
                    "(set -p out:=<path> or ensure map_topic is published)")
                return
            out_path = os.path.join(self.map_dir, self.get_parameter("align_filename").value)
        if not map_name and self.map_dir:
            map_name = os.path.basename(os.path.normpath(self.map_dir))

        paired = self._pair()
        if paired is None or len(paired[0]) == 0:
            self.get_logger().error("no paired samples; nothing to fit")
            return
        X, lla, stat = paired
        fixed = np.where(stat >= NavSatStatus.STATUS_GBAS_FIX)[0]
        min_pairs = int(self.get_parameter("min_pairs").value)
        if len(fixed) < min_pairs:
            self.get_logger().error(
                f"only {len(fixed)} RTK-FIXED pairs (< min_pairs={min_pairs}); "
                "collect more / get a better fix")
            return
        o = lla[fixed[0]]
        frame = rtk_geo.EnuFrame(o[0], o[1], o[2])
        enu = np.array([frame.lla_to_enu(a, b, c) for a, b, c in lla])
        gross = float(self.get_parameter("gross_reject_m").value)
        yaw, s, t2, mask, rmse = rtk_geo.robust_sim3(X[fixed], enu[fixed][:, :2], gross)

        # Local-weighted correspondence cloud: the global Sim3 is limited to
        # ~1.5 m by the VIO map warp; storing the map<->ENU pairs lets the runtime
        # fit a Sim3 locally around each fix (~0.3 m). Keep every fixed pair whose
        # residual to the global fit is under `local_keep_m` (drop only the >5 m
        # gross blunders, e.g. pre-convergence reloc jumps); the rest are all
        # valid points to fit.
        Xf = X[fixed][:, :2]
        Ef = enu[fixed][:, :2]
        keep_m = float(self.get_parameter("local_keep_m").value)
        rr = rtk_geo.sim3_resid(yaw, s, t2, Xf, Ef)
        pmask = rr <= keep_m
        local_bw = float(self.get_parameter("local_bw_m").value)

        out = {
            "map": map_name or None,
            "model": "local_weighted_sim3",
            "convention": ("p_enu = scale * R(yaw_deg) * p_map_xy + [tx,ty]; "
                           "inverse: p_map_xy = (1/scale) * R(yaw_deg)^T * (p_enu_xy - [tx,ty]). "
                           "Runtime prefers local-weighted Sim3 over 'points'; 'sim3' is the global fallback."),
            "origin_lla": {"lat": float(o[0]), "lon": float(o[1]), "alt": float(o[2])},
            "sim3": {"yaw_deg": float(np.degrees(yaw)), "scale": float(s),
                     "tx": float(t2[0]), "ty": float(t2[1])},
            "local": {"model": "local_weighted_sim3", "bw_m": local_bw,
                      "min_neighbors": 15, "bw_max_m": 40.0},
            "points": {"map_xy": np.round(Xf[pmask], 4).tolist(),
                       "enu_xy": np.round(Ef[pmask], 4).tolist()},
            "orientation": "position-only (single antenna); use VIO orientation",
            "fit": {"n_used": int(mask.sum()), "n_fixed_pairs": int(len(fixed)),
                    "n_all_pairs": int(len(X)), "rmse_m": round(float(rmse), 3),
                    "gross_reject_m": gross, "n_local_points": int(pmask.sum()),
                    "local_keep_m": keep_m},
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        self.get_logger().info(
            f"WROTE {out_path}: yaw={np.degrees(yaw):.2f} scale={s:.4f} "
            f"rmse={rmse:.2f}m used={int(mask.sum())}/{len(fixed)}")


def main():
    rclpy.init()
    node = RtkAlignCalibrate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("stopping, fitting…")
    finally:
        node.fit_and_save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

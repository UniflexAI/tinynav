#!/usr/bin/env python3
"""Runtime: turn live RTK (+ IMU) into the robot's pose in the saved map.

Converts each /fix (lat/lon) to map-frame XY via the inverse planar Sim3 stored
in a per-map rtk_align.json (produced by rtk_align_calibrate.py), and publishes
/rtk/map_pose (nav_msgs/Odometry, map frame) with position AND heading.

Heading (see rtk_heading.py):
  Single-antenna RTK has no direct heading; course-over-ground only works while
  driving straight forward and is useless during spot turns / reverse — exactly
  what a navigating robot does. So heading is a fusion:
    * high-rate IMU gyro   -> responsive heading (tracks turns / reverse),
    * low-rate RTK track    -> absolute drift-free heading, but only sampled when
                               the recent travel is straight enough.
  The IMU carries the heading through turns; each straight RTK segment corrects
  the gyro's slow drift and its bias. First straight segment also bootstraps the
  absolute heading (the NEED_YAW_INIT -> ACTIVE handshake).

Map gating (which rtk_align.json to load):
  This node does NOT hard-code a map. It subscribes to `map_topic` (std_msgs/
  String = the current map DIRECTORY path) and loads `<map_dir>/rtk_align.json`
  from it, mirroring how map_node reads nav_flow.json from the same directory.
  It only publishes /rtk/map_pose once a map with an rtk_align.json is active.

  IMPORTANT (integration contract): publishing `map_topic` is the map/navigation
  owner's job, NOT part of the RTK module. That publisher must:
    - publish a std_msgs/String whose data is the map directory (the same path
      passed to map_node as --tinynav_map_path), and
    - use a LATCHED QoS (durability=TRANSIENT_LOCAL, depth 1) so this node gets
      the current map even if it starts late.
  For bench testing you can bypass the topic with -p align_json:=<path>.

Decoupled: interacts with the rest of the stack only via ROS topics + the json.

Usage (normal, topic-gated):
  uv run python /tinynav/rtk/rtk_map_pose_node.py --ros-args \
      -p map_topic:=/map/current_map
Usage (bench, fixed json, no topic):
  uv run python /tinynav/rtk/rtk_map_pose_node.py --ros-args \
      -p align_json:=<tinynav_map_path>/rtk_align.json
"""
import json
import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       HistoryPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtk_geo
from rtk_heading import (StraightYaw, HeadingFilter, OdomOffsetHeading,
                         yaw_from_odom_quat)


class RtkMapPoseNode(Node):
    def __init__(self):
        super().__init__("rtk_map_pose_node")
        # Map source: subscribe to the current-map topic (owner publishes it) and
        # load <map_dir>/<align_filename>. align_json is an optional bench bypass.
        self.declare_parameter("map_topic", "/map/current_map")
        self.declare_parameter("align_filename", "rtk_align.json")
        self.declare_parameter("align_json", "")
        self.declare_parameter("fix_topic", "/fix")
        # Heading relative source: "odom" (VIO yaw from /slam/odometry, default)
        # or "imu" (integrate /lidar_imu gyro). Both are anchored to the map by
        # the same straight-segment RTK correction.
        self.declare_parameter("heading_source", "odom")
        self.declare_parameter("imu_topic", "/lidar_imu")
        self.declare_parameter("odom_topic", "/slam/odometry")
        self.declare_parameter("heading_odom_alpha", 0.2)  # odom->map offset LP
        # False (default) = heading is pure odom + a one-time initial offset
        # (offset bootstrapped from the first straight RTK segment, then locked;
        # RTK no longer corrects heading). True = keep low-pass RTK correction.
        self.declare_parameter("heading_odom_correct", False)
        self.declare_parameter("output_topic", "/rtk/map_pose")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("child_frame_id", "rtk")
        # Publish only at RTK FIXED. rtk_bridge_node maps both GGA q4/q5 to
        # NavSatStatus.STATUS_GBAS_FIX, so use its covariance model to keep q5
        # RTK_FLOAT out: q4 sigma_h~=0.02m, q5 sigma_h~=0.15m.
        self.declare_parameter("min_status", int(NavSatStatus.STATUS_GBAS_FIX))
        self.declare_parameter("fixed_max_horizontal_sigma_m", 0.05)
        # RTK straight-segment heading observation (course-over-ground).
        self.declare_parameter("heading_min_dist_m", 1.0)   # window length
        self.declare_parameter("straight_max_offline_m", 0.15)  # straightness gate
        # IMU / heading fusion.
        self.declare_parameter("heading_kp", 0.2)           # angle pull per obs
        self.declare_parameter("heading_ki", 0.02)          # bias learn per obs
        self.declare_parameter("gravity_tau_s", 1.0)        # gravity low-pass
        self.declare_parameter("yaw_rate_sign", 1.0)        # flip if turns invert
        # Yaw covariance model (deg): base + drift growth since last correction.
        self.declare_parameter("yaw_base_std_deg", 3.0)
        self.declare_parameter("yaw_drift_deg_per_s", 0.05)
        # Handshake status topic (std_msgs/String JSON).
        self.declare_parameter("status_topic", "/rtk/init_status")
        self.declare_parameter("status_rate_hz", 2.0)
        self.declare_parameter("fix_timeout_s", 2.0)

        self.align_filename = self.get_parameter("align_filename").value
        self.min_status = int(self.get_parameter("min_status").value)
        self.fixed_max_horizontal_sigma_m = float(self.get_parameter("fixed_max_horizontal_sigma_m").value)
        self.map_frame = self.get_parameter("map_frame_id").value
        self.child_frame = self.get_parameter("child_frame_id").value
        self.fix_timeout_s = float(self.get_parameter("fix_timeout_s").value)
        self.yaw_base_std_deg = float(self.get_parameter("yaw_base_std_deg").value)
        self.yaw_drift_deg_per_s = float(self.get_parameter("yaw_drift_deg_per_s").value)

        # Not loaded until a map is resolved. on_fix stays silent until then.
        self.loaded_path = None
        self.meta = self.enu = self.yaw = self.scale = self.t2 = None
        self.pts_map = self.pts_enu = None     # local-weighted correspondence cloud
        self.local_bw = 5.0
        self.local_min_pts = 15
        self.local_bw_max = 40.0
        # Heading fusion state.
        self.sy = StraightYaw(
            min_dist=float(self.get_parameter("heading_min_dist_m").value),
            max_offline_m=float(self.get_parameter("straight_max_offline_m").value))
        self.heading_source = str(self.get_parameter("heading_source").value).lower()
        if self.heading_source == "imu":
            self.hf = HeadingFilter(
                kp=float(self.get_parameter("heading_kp").value),
                ki=float(self.get_parameter("heading_ki").value),
                g_tau=float(self.get_parameter("gravity_tau_s").value),
                yaw_rate_sign=float(self.get_parameter("yaw_rate_sign").value))
        else:
            self.heading_source = "odom"
            self.hf = OdomOffsetHeading(
                alpha=float(self.get_parameter("heading_odom_alpha").value),
                correct=bool(self.get_parameter("heading_odom_correct").value))
        self.last_xy = None              # latest map-frame position (x, y)
        self.last_imu_t = None           # last IMU header time (s), for dt
        # Latest-fix bookkeeping for the handshake status.
        self.last_status = -1
        self.last_fix_is_fixed = False
        self.last_fix_wall = None

        self.pub = self.create_publisher(
            Odometry, self.get_parameter("output_topic").value, 10)
        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10)
        self.create_subscription(
            NavSatFix, self.get_parameter("fix_topic").value, self.on_fix, 20)
        if self.heading_source == "imu":
            self.create_subscription(
                Imu, self.get_parameter("imu_topic").value, self.on_imu,
                qos_profile_sensor_data)
        else:
            self.create_subscription(
                Odometry, self.get_parameter("odom_topic").value, self.on_odom, 20)
        rate = float(self.get_parameter("status_rate_hz").value)
        self.create_timer(1.0 / rate if rate > 0 else 1.0, self._publish_status)

        bench_json = self.get_parameter("align_json").value
        if bench_json:
            # Bench bypass: load a fixed json directly, ignore the map topic.
            if not self._load(bench_json):
                raise SystemExit(f"align_json not found/invalid: {bench_json!r}")
        else:
            # Latched sub so a map published before we started is still delivered.
            latched = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                String, self.get_parameter("map_topic").value, self.on_map, latched)
            self.get_logger().info(
                f"rtk_map_pose waiting for map on "
                f"{self.get_parameter('map_topic').value!r} "
                "(publishes /rtk/map_pose once a map with rtk_align.json is active)")

    # ---- map gating -------------------------------------------------------
    def _reset_heading(self):
        self.sy.reset()
        self.hf.reset()
        self.last_xy = None

    def _load(self, path):
        """Load an rtk_align.json; return True on success."""
        if not path or not os.path.isfile(path):
            return False
        try:
            self.meta, self.enu, self.yaw, self.scale, self.t2 = rtk_geo.load_align(path)
        except Exception as exc:
            self.get_logger().error(f"failed to load {path}: {exc}")
            return False
        # Optional local-weighted correspondence cloud: when present, the fix->map
        # conversion uses a Sim3 fit locally around each fix (absorbs the VIO map
        # warp a single global Sim3 cannot), falling back to the global Sim3.
        self.pts_map = self.pts_enu = None
        pts = self.meta.get("points") if isinstance(self.meta, dict) else None
        if isinstance(pts, dict) and pts.get("map_xy") and pts.get("enu_xy"):
            try:
                pm = np.asarray(pts["map_xy"], float)
                pe = np.asarray(pts["enu_xy"], float)
                if pm.ndim == 2 and pm.shape == pe.shape and len(pm) >= 5:
                    self.pts_map, self.pts_enu = pm, pe
            except Exception as exc:
                self.get_logger().warning(f"ignoring align 'points': {exc}")
        lc = self.meta.get("local") if isinstance(self.meta, dict) else None
        lc = lc if isinstance(lc, dict) else {}
        self.local_bw = float(lc.get("bw_m", 5.0))
        self.local_min_pts = int(lc.get("min_neighbors", 15))
        self.local_bw_max = float(lc.get("bw_max_m", 40.0))
        self._reset_heading()            # new map frame -> re-bootstrap heading
        self.loaded_path = path
        self.get_logger().info(
            f"loaded rtk_align: map={self.meta.get('map')} path={path} "
            f"yaw={math.degrees(self.yaw):.1f} scale={self.scale:.4f} "
            f"origin=({self.enu.lat:.7f},{self.enu.lon:.7f}) "
            f"local={'on(%d pts,bw=%.1f)' % (len(self.pts_map), self.local_bw) if self.pts_map is not None else 'off'}")
        return True

    def _clear_active_map(self, reason):
        had_map = self.enu is not None or self.loaded_path is not None
        self.loaded_path = None
        self.meta = self.enu = self.yaw = self.scale = self.t2 = None
        self.pts_map = self.pts_enu = None
        self._reset_heading()
        if had_map:
            self.get_logger().info(f"cleared active RTK map: {reason}")

    def on_map(self, msg: String):
        map_dir = (msg.data or "").strip()
        if not map_dir:
            self._clear_active_map("empty map topic")
            return
        path = os.path.join(map_dir, self.align_filename)
        if path == self.loaded_path:
            return                       # already active, nothing to do
        if not os.path.isfile(path):
            self._clear_active_map(f"missing {self.align_filename} in {map_dir!r}")
            self.get_logger().warning(
                f"map {map_dir!r} has no {self.align_filename} "
                "(not RTK-calibrated); /rtk/map_pose paused")
            return
        self._load(path)

    # ---- sensors ----------------------------------------------------------
    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _is_fixed_fix(self, msg: NavSatFix) -> bool:
        if int(msg.status.status) < self.min_status:
            return False
        cov = list(msg.position_covariance)
        if len(cov) < 8:
            return False
        sigma_x = math.sqrt(max(float(cov[0]), 0.0))
        sigma_y = math.sqrt(max(float(cov[7]), 0.0))
        return max(sigma_x, sigma_y) <= self.fixed_max_horizontal_sigma_m

    def on_imu(self, msg: Imu):
        """Advance the heading with the gyro (gravity-projected). High rate."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if 0.0 < dt < 0.5:           # ignore gaps / out-of-order stamps
                self.hf.imu_update(
                    (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z),
                    (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z),
                    dt)
        self.last_imu_t = t

    def on_odom(self, msg: Odometry):
        """Feed the odometry (VIO) heading as the relative source. High rate."""
        q = msg.pose.pose.orientation
        self.hf.set_odom_yaw(yaw_from_odom_quat(q.x, q.y, q.z, q.w))

    def on_fix(self, msg: NavSatFix):
        # Record quality/time first (drives the handshake status even at low fix).
        self.last_status = int(msg.status.status)
        self.last_fix_is_fixed = self._is_fixed_fix(msg)
        self.last_fix_wall = self._now_s()
        if self.enu is None:                      # gated: no active map yet
            return
        if not self.last_fix_is_fixed:
            return
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return
        enu = self.enu.lla_to_enu(msg.latitude, msg.longitude, msg.altitude)
        if self.pts_map is not None:
            xy, _, _ = rtk_geo.enu_to_map_xy_local(
                self.pts_map, self.pts_enu, enu[:2],
                bw=self.local_bw, min_pts=self.local_min_pts, bw_max=self.local_bw_max,
                fallback=(self.yaw, self.scale, self.t2))
        else:
            xy = rtk_geo.enu_to_map_xy(self.yaw, self.scale, self.t2, enu[:2])
        self.last_xy = (float(xy[0]), float(xy[1]))
        # Straight-segment heading observation -> bootstrap / correct the filter.
        obs = self.sy.add(self.last_xy[0], self.last_xy[1])
        if obs is not None:
            self.hf.rtk_observe(obs[0], self._now_s())
        # Publish the pose ONCE PER FIX so the position is always fresh: the RTK
        # position only changes at the /fix rate (~1 Hz), so emitting faster would
        # just repeat the same point and mislead a consumer into "not moving".
        # The heading in each message is the current fused yaw (the IMU has kept
        # it correct through any turns/reverse between fixes).
        if self.hf.ready:
            self._publish_pose(msg.header.stamp)

    # ---- outputs ----------------------------------------------------------
    def _publish_pose(self, stamp):
        now = self._now_s()
        half = self.hf.heading / 2.0
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.map_frame
        od.child_frame_id = self.child_frame
        od.pose.pose.position.x = self.last_xy[0]
        od.pose.pose.position.y = self.last_xy[1]
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.z = math.sin(half)   # yaw about map +Z
        od.pose.pose.orientation.w = math.cos(half)
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.25         # x, y (RTK FIXED/FLOAT, cm-level)
        cov[14] = 1e6                  # z (2D fit, unknown)
        cov[21] = cov[28] = 1e6        # roll, pitch unknown
        cov[35] = self.hf.yaw_var(now, self.yaw_base_std_deg,
                                  self.yaw_drift_deg_per_s)
        od.pose.covariance = cov
        self.pub.publish(od)

    def _publish_status(self):
        """Continuously publish the init handshake state for the nav node.

        States:
          NO_MAP        - no active map / no rtk_align.json -> nothing to do
          WAIT_FIX      - map ready, waiting for RTK FIXED
          NEED_YAW_INIT - map + RTK FIXED but heading not bootstrapped -> DRIVE ~1 m
          ACTIVE        - heading acquired; /rtk/map_pose is publishing
        The nav node should drive the robot slowly forward (with its own obstacle
        avoidance) while need_forward_init is true, and stop once ACTIVE.
        """
        have_map = self.enu is not None
        fix_recent = (self.last_fix_wall is not None
                      and self._now_s() - self.last_fix_wall < self.fix_timeout_s)
        fix_ok = fix_recent and self.last_fix_is_fixed
        yaw_ready = self.hf.ready
        if not have_map:
            state = "NO_MAP"
        elif not fix_ok:
            state = "WAIT_FIX"
        elif not yaw_ready:
            state = "NEED_YAW_INIT"
        else:
            state = "ACTIVE"
        payload = {
            "state": state,
            "need_forward_init": state == "NEED_YAW_INIT",
            "have_map": have_map,
            "map": (self.meta or {}).get("map") if have_map else None,
            "fix_ok": bool(fix_ok),
            "navsat_status": self.last_status,   # 2 == GBAS_FIX (RTK q4/5)
            "fixed_max_horizontal_sigma_m": self.fixed_max_horizontal_sigma_m,
            "yaw_ready": yaw_ready,
            "yaw_deg": None if not yaw_ready else round(math.degrees(self.hf.heading), 1),
            "heading_source": self.heading_source,
        }
        self.status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"))))


def main():
    rclpy.init()
    node = RtkMapPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

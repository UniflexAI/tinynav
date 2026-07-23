#!/usr/bin/env python3
"""Runtime: turn live RTK into the robot's position in the saved map.

Converts each /fix (lat/lon) to map-frame XY via the inverse planar Sim3 stored
in a per-map rtk_align.json (produced by rtk_align_calibrate.py), and publishes
/rtk/map_pose (nav_msgs/Odometry, map frame).

Position only: single-antenna RTK has no usable heading, so orientation is left
identity for a downstream consumer to fill (VIO / motion-fit). The pose
covariance encodes RTK quality (tight at RTK FIXED/FLOAT, loose at DGNSS) so the
consumer can decide how to use it (primary at FIXED, only a search-range
constraint at DGNSS).

Map gating (which rtk_align.json to load):
  This node does NOT hard-code a map. It subscribes to `map_topic` (std_msgs/
  String = the current map DIRECTORY path) and loads `<map_dir>/rtk_align.json`
  from it, mirroring how map_node reads nav_flow.json from the same directory.
  It only publishes /rtk/map_pose once a map with an rtk_align.json is active;
  if the map has no rtk_align.json (not RTK-calibrated) it stays silent.

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
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rtk_geo


class RtkMapPoseNode(Node):
    def __init__(self):
        super().__init__("rtk_map_pose_node")
        # Map source: subscribe to the current-map topic (owner publishes it) and
        # load <map_dir>/<align_filename>. align_json is an optional bench bypass.
        self.declare_parameter("map_topic", "/map/current_map")
        self.declare_parameter("align_filename", "rtk_align.json")
        self.declare_parameter("align_json", "")
        self.declare_parameter("fix_topic", "/fix")
        self.declare_parameter("output_topic", "/rtk/map_pose")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("child_frame_id", "rtk")
        # Publish only at RTK FIXED/FLOAT (q4/5): cm-level fixes. DGNSS/single are
        # too noisy both for the position and for the motion-fit heading.
        self.declare_parameter("min_status", int(NavSatStatus.STATUS_GBAS_FIX))
        # Heading from RTK motion (course-over-ground): map-frame yaw = direction
        # of travel of the map-frame track over the last heading_min_dist_m. No
        # IMU/VIO, so yaw is only trustworthy while moving forward; its covariance
        # inflates once the last fit goes stale (robot stopped / may have turned
        # in place, which we cannot observe without an inertial source).
        self.declare_parameter("heading_min_dist_m", 1.0)
        self.declare_parameter("yaw_std_deg", 5.0)
        self.declare_parameter("heading_stale_s", 3.0)
        # Handshake status topic (std_msgs/String JSON), published continuously so
        # the navigation node knows when to drive the ~1 m yaw-init forward motion.
        self.declare_parameter("status_topic", "/rtk/init_status")
        self.declare_parameter("status_rate_hz", 2.0)
        self.declare_parameter("fix_timeout_s", 2.0)

        self.align_filename = self.get_parameter("align_filename").value
        self.min_status = int(self.get_parameter("min_status").value)
        self.map_frame = self.get_parameter("map_frame_id").value
        self.child_frame = self.get_parameter("child_frame_id").value
        self.heading_min_dist = float(self.get_parameter("heading_min_dist_m").value)
        self.yaw_var_fresh = math.radians(float(self.get_parameter("yaw_std_deg").value)) ** 2
        self.heading_stale_s = float(self.get_parameter("heading_stale_s").value)

        # Not loaded until a map is resolved. on_fix stays silent until then.
        self.loaded_path = None
        self.meta = self.enu = self.yaw = self.scale = self.t2 = None
        # Course-over-ground heading state (map frame).
        self.track = deque(maxlen=400)   # recent map-frame (x, y) at q4/5
        self.yaw_est = None              # last fitted map-frame heading (rad)
        self.last_fit_wall = None        # wall time (s) of the last heading fit
        # Latest-fix bookkeeping for the handshake status.
        self.last_status = -1            # last NavSatStatus.status seen
        self.last_fix_wall = None        # wall time (s) of the last fix
        self.fix_timeout_s = float(self.get_parameter("fix_timeout_s").value)

        self.pub = self.create_publisher(
            Odometry, self.get_parameter("output_topic").value, 10)
        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10)
        self.create_subscription(
            NavSatFix, self.get_parameter("fix_topic").value, self.on_fix, 20)
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

    def _reset_heading(self):
        self.track.clear()
        self.yaw_est = None
        self.last_fit_wall = None

    def _load(self, path):
        """Load an rtk_align.json; return True on success."""
        if not path or not os.path.isfile(path):
            return False
        try:
            self.meta, self.enu, self.yaw, self.scale, self.t2 = rtk_geo.load_align(path)
        except Exception as exc:
            self.get_logger().error(f"failed to load {path}: {exc}")
            return False
        self._reset_heading()            # new map frame -> re-bootstrap heading
        self.loaded_path = path
        self.get_logger().info(
            f"loaded rtk_align: map={self.meta.get('map')} path={path} "
            f"yaw={np.degrees(self.yaw):.1f} scale={self.scale:.4f} "
            f"origin=({self.enu.lat:.7f},{self.enu.lon:.7f})")
        return True

    def on_map(self, msg: String):
        map_dir = (msg.data or "").strip()
        if not map_dir:
            return
        path = os.path.join(map_dir, self.align_filename)
        if path == self.loaded_path:
            return                       # already active, nothing to do
        if not os.path.isfile(path):
            # Map switched to one without a calibration -> stop publishing.
            self.loaded_path = None
            self.meta = self.enu = self.yaw = self.scale = self.t2 = None
            self._reset_heading()
            self.get_logger().warning(
                f"map {map_dir!r} has no {self.align_filename} "
                "(not RTK-calibrated); /rtk/map_pose paused")
            return
        self._load(path)

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _update_heading(self, x, y):
        """Map-frame yaw from a least-squares line fit over the last
        heading_min_dist_m of the map-frame track (course-over-ground). A line
        fit over all points in the window is far less sensitive to per-fix noise
        than a single two-point chord, so a shorter window still gives a stable
        yaw. Sign is resolved by net travel so the heading points forward."""
        self.track.append((x, y))
        # Window = newest back to the first point >= heading_min_dist away.
        min_d2 = self.heading_min_dist ** 2
        start = None
        for i in range(len(self.track) - 1, -1, -1):
            px, py = self.track[i]
            if (x - px) ** 2 + (y - py) ** 2 >= min_d2:
                start = i
                break
        if start is None:
            return                                   # < heading_min_dist so far
        window = list(self.track)[start:]            # oldest -> newest, spans ~1 m
        if len(window) < 2:
            return
        P = np.asarray(window, dtype=float)
        net = P[-1] - P[0]                           # net travel (sign reference)
        c = P - P.mean(axis=0)
        _, vecs = np.linalg.eigh(c.T @ c)            # ascending eigenvalues
        axis = vecs[:, -1]                           # principal (line) direction
        if float(axis @ net) < 0.0:                  # orient along travel
            axis = -axis
        self.yaw_est = math.atan2(float(axis[1]), float(axis[0]))
        self.last_fit_wall = self._now_s()

    def on_fix(self, msg: NavSatFix):
        # Record quality/time first (drives the handshake status even at low fix).
        self.last_status = int(msg.status.status)
        self.last_fix_wall = self._now_s()
        if self.enu is None:                 # gated: no active map yet
            return
        if msg.status.status < self.min_status:   # q4/5 only
            return
        if not (np.isfinite(msg.latitude) and np.isfinite(msg.longitude)):
            return
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return
        enu = self.enu.lla_to_enu(msg.latitude, msg.longitude, msg.altitude)
        xy = rtk_geo.enu_to_map_xy(self.yaw, self.scale, self.t2, enu[:2])

        self._update_heading(float(xy[0]), float(xy[1]))
        if self.yaw_est is None:
            # Position + orientation are published together; hold until the robot
            # has moved enough for a first heading fit.
            return

        # yaw covariance: fresh right after a motion fit, inflated once stale
        # (robot may have stopped or turned in place — unobservable here).
        age = self._now_s() - self.last_fit_wall
        yaw_var = self.yaw_var_fresh if age <= self.heading_stale_s else math.radians(90.0) ** 2
        half = self.yaw_est / 2.0

        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = self.map_frame
        od.child_frame_id = self.child_frame
        od.pose.pose.position.x = float(xy[0])
        od.pose.pose.position.y = float(xy[1])
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation.z = math.sin(half)   # yaw about map +Z
        od.pose.pose.orientation.w = math.cos(half)

        cov = [0.0] * 36
        cov[0] = cov[7] = 0.25     # x, y (RTK FIXED/FLOAT, cm-level)
        cov[14] = 1e6              # z (2D fit, unknown)
        cov[21] = cov[28] = 1e6    # roll, pitch unknown
        cov[35] = yaw_var          # yaw from motion fit
        od.pose.covariance = cov
        self.pub.publish(od)

    def _publish_status(self):
        """Continuously publish the init handshake state for the nav node.

        States:
          NO_MAP        - no active map / no rtk_align.json -> nothing to do
          WAIT_FIX      - map ready, waiting for RTK FIXED/FLOAT (q4/5)
          NEED_YAW_INIT - map + q4/5 but no heading yet -> DRIVE FORWARD ~1 m
          ACTIVE        - heading acquired; /rtk/map_pose is publishing
        The nav node should drive the robot slowly forward (with its own obstacle
        avoidance) while need_forward_init is true, and stop once ACTIVE.
        """
        have_map = self.enu is not None
        fix_recent = (self.last_fix_wall is not None
                      and self._now_s() - self.last_fix_wall < self.fix_timeout_s)
        fix_ok = fix_recent and self.last_status >= self.min_status  # q4/5
        yaw_ready = self.yaw_est is not None
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
            "yaw_ready": yaw_ready,
            "yaw_deg": None if not yaw_ready else round(math.degrees(self.yaw_est), 1),
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

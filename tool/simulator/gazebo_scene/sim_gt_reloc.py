#!/usr/bin/env python3
"""Ground-truth relocalization correction for the tinynav gz (Fortress) sim.

Without map_node there is no visual relocalization, and perception_node has no
pose-reset interface: its SLAM world anchors to identity at perception startup
(A) and drifts over time (E), so /slam/odometry_visual lives in a frame that
only coincides with the gazebo world by luck. The old workaround respawned the
perception tmux window per scene run so SLAM re-anchored at the origin -- slow
(TRT reload) and racy. This node replaces that hack.

It consumes the raw SLAM odometry (perception is launched with
-r /slam/odometry_visual:=/slam/odometry_visual_raw), compares it against the
gazebo ground truth of the robot chassis (bridged from /world/$WORLD/
pose_info as gz.msgs.Pose_V -> tf2_msgs/TFMessage), and republishes the
odometry corrected into the gazebo world frame on the original topic name:

    P(t) = A . X(t) . E(t)        raw SLAM output
    Q(t) = T_gt(t) . C_hat        published correction

where X(t) is the true camera pose. C_hat (chassis -> SLAM output frame) is a
physical constant (camera extrinsics in the SLAM output convention), so Q is
exact ground truth up to that constant frame -- same convention the raw SLAM
establishes at its anchor, minus anchor instability and drift: E(t) is divided
out because Q never depends on the quality of P, only on its existence, which
makes mid-run teleports (set_pose back to the origin) harmless.

Calibration: perception anchors its first keyframe at a zero-translation pose
([R_anchor | 0]), so the earliest odometry samples satisfy
C_hat = T_gt(t)^-1 . P(t) with translation(P) == 0. Calibration must happen
in that early window, before graph optimization wanders the anchor (observed
to reach >0.1m / 120deg while the robot stares at a textureless floor during
the ~1min TRT engine load). This node therefore tracks, over the first
CALIB_WINDOW_S of raw odometry, the sample whose translation is closest to
zero, requires the robot to be stationary (checked against ground truth), and
-- since the anchor rotation R_anchor varies between runs (gravity-align
degeneracy) -- checks translation only, never rotation.

If the window ends with no sample close enough to the anchor (robot moved
during perception init, or the graph already wandered), calibration is
refused, retried, and scene_runner eventually times out with a clear message;
respawning the perception window restores a fresh anchor.

If /map/relocalization gains a publisher (map_node running), the node falls
back to pass-through so map_node's own relocalization stays the sole authority
(run_simulator.sh --map does not even start this node; this is the belt for
manually started map_nodes).

Calibration readiness is published latched on /sim/gt_reloc/status (Bool);
scene_runner waits for it before touching the robot.
"""

import argparse
import subprocess
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

from tinynav.core.math_utils import msg2np, np2msg, tf2np

CALIB_WINDOW_S = 30.0       # track the best (min |translation|) sample this long
CALIB_OK_TOL = 0.01         # calibrate immediately below this |translation|
CALIB_ACCEPT_TOL = 0.05     # accept the window's best sample below this
GT_STATIONARY_TOL = 0.003   # max GT translation spread while calibrating
GT_STALE_S = 0.5            # passthrough if the ground truth is older than this


def discover_world(timeout=180.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = subprocess.run(["ign", "topic", "-l"], capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("/world/") and line.endswith("/pose/info"):
                return line.split("/")[2]
        time.sleep(2.0)
    return None


class SimGtReloc(Node):
    def __init__(self, world, robot_name):
        super().__init__("sim_gt_reloc")
        self.robot_name = robot_name
        self.bypass = False
        self.calibrated = False
        self.C_hat = None
        self.gt_T = None
        self.gt_wall = 0.0
        self.gt_hist = []  # (monotonic, translation) for stationarity checks
        self.raw_msg = None
        self._logged_names = False
        self._last_stale_warn = 0.0
        self._last_refuse_warn = 0.0
        self._first_raw_wall = None
        self._best = None  # (|translation|, gt_T, P) of best sample in window

        self.create_subscription(
            TFMessage, f"/world/{world}/pose/info", self.gt_cb,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Odometry, "/slam/odometry_visual_raw", self.raw_cb, 50)
        self.pub = self.create_publisher(Odometry, "/slam/odometry_visual", 50)
        self.status_pub = self.create_publisher(
            Bool, "/sim/gt_reloc/status",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(0.2, self.calib_tick)
        self.create_timer(2.0, self.guard_tick)
        self.publish_status(False)
        self.get_logger().info(
            f"sim_gt_reloc up (world={world}, robot={robot_name}); waiting for "
            "perception's first odometry to calibrate")

    # ---- subscriptions ----

    def gt_cb(self, msg):
        if not self._logged_names:
            self._logged_names = True
            self.get_logger().info(
                "pose/info entities: " + ", ".join(t.child_frame_id for t in msg.transforms))
        now = time.monotonic()
        for t in msg.transforms:
            if t.child_frame_id.split("/")[-1] != self.robot_name:
                continue
            _, _, T = tf2np(t)
            self.gt_T = T
            self.gt_wall = now
            self.gt_hist.append((now, T[:3, 3].copy()))
            if len(self.gt_hist) > 200:
                self.gt_hist = self.gt_hist[-100:]

    def raw_cb(self, msg):
        self.raw_msg = msg
        if not self.calibrated and not self.bypass:
            T, _ = msg2np(msg)
            self.track_calibration_sample(T)
        else:
            self.publish_corrected()

    # ---- calibration ----

    def gt_stationary(self):
        now = time.monotonic()
        recent = [p for w, p in self.gt_hist if now - w < 1.0]
        if len(recent) < 5:
            return False
        return max(np.linalg.norm(p - recent[0]) for p in recent) < GT_STATIONARY_TOL

    def track_calibration_sample(self, P):
        now = time.monotonic()
        if self._first_raw_wall is None:
            self._first_raw_wall = now
        if self.gt_T is None or not self.gt_stationary():
            return
        t_norm = float(np.linalg.norm(P[:3, 3]))
        if self._best is None or t_norm < self._best[0]:
            self._best = (t_norm, self.gt_T.copy(), P.copy())
        if t_norm < CALIB_OK_TOL:
            self.calibrate(*self._best)
        elif now - self._first_raw_wall > CALIB_WINDOW_S:
            if self._best[0] < CALIB_ACCEPT_TOL:
                self.calibrate(*self._best)
            elif now - self._last_refuse_warn > 10.0:
                self._last_refuse_warn = now
                self.get_logger().warn(
                    f"calibration refused: best |translation| in window was "
                    f"{self._best[0]:.3f}m (> {CALIB_ACCEPT_TOL}m) -- the SLAM "
                    "anchor already wandered or the robot moved during "
                    "perception init; respawn the perception window to restore "
                    "a fresh anchor (retrying meanwhile)")

    def calib_tick(self):
        # close the window even if no further samples arrive
        if self.calibrated or self.bypass or self._first_raw_wall is None or self._best is None:
            return
        if time.monotonic() - self._first_raw_wall > CALIB_WINDOW_S and self._best[0] < CALIB_ACCEPT_TOL:
            self.calibrate(*self._best)

    def calibrate(self, t_norm, gt_T, P):
        self.C_hat = np.linalg.inv(gt_T) @ P
        self.calibrated = True
        self.get_logger().info(
            f"calibrated: |P translation| = {t_norm:.4f}m, C_hat t = "
            f"{np.round(self.C_hat[:3, 3], 4)}; publishing corrected odometry")
        self.publish_status(True)

    def guard_tick(self):
        if self.bypass:
            return
        if self.count_publishers("/map/relocalization") > 0:
            self.bypass = True
            self.calibrated = True  # passthrough counts as ready
            self.get_logger().warn(
                "/map/relocalization is published (map_node running) -- passing "
                "raw odometry through untouched")
            self.publish_status(True)

    # ---- output ----

    def publish_corrected(self):
        if self.bypass:
            self.pub.publish(self.raw_msg)
            return
        now = time.monotonic()
        if self.gt_T is None or now - self.gt_wall > GT_STALE_S:
            if now - self._last_stale_warn > 5.0:
                self._last_stale_warn = now
                self.get_logger().warn("ground truth stale; passing raw odometry through")
            self.pub.publish(self.raw_msg)
            return
        Q = self.gt_T @ self.C_hat
        out = np2msg(Q, self.raw_msg.header.stamp,
                     self.raw_msg.header.frame_id or "world",
                     self.raw_msg.child_frame_id or "camera")
        # twist is body-frame velocity; the world-frame correction leaves it valid
        out.twist = self.raw_msg.twist
        out.pose.covariance = self.raw_msg.pose.covariance
        self.pub.publish(out)

    def publish_status(self, ready):
        m = Bool()
        m.data = bool(ready)
        self.status_pub.publish(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-name", default="lekiwi")
    ap.add_argument("--world", default=None, help="gz world name (default: auto-discover)")
    args = ap.parse_args()

    world = args.world or discover_world()
    if world is None:
        raise SystemExit("no gz world found (is the sim running?)")
    rclpy.init(args=[])
    node = SimGtReloc(world, args.robot_name)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

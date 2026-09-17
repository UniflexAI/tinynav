#!/usr/bin/env python3
"""gz ground-truth goal transformer for the tinynav gz (Fortress) sim.

planning lives in perception's raw SLAM world frame ("world", the TF root) and
its odometry input is perception's raw stream (the launcher remaps
/slam/odometry_visual to it). Scene goals, however, are authored in gazebo
world coordinates (scene JSON), a frame that only coincides with the SLAM
frame by luck. This node keeps the two glued at goal-publishing time:

    g_slam(t) = P(t) . E^-1 . T_gt(t)^-1 . g_gz

P(t) is perception's raw odometry (/slam/odometry_visual_raw), T_gt(t) the
robot ground truth (bridged /world/$WORLD/pose/info), E the static camera
extrinsic from the lekiwi model origin to the perception camera frame (from
tool/simulator/worlds/factory_scene.sdf: chassis sits (0,0,0.083) above the
model origin, infra1_link (0.09,0.0255,0.017) in chassis; the rotation is the
camera axes -- x right, y down, z fwd -- expressed in the chassis frame).

Everything is computed from the current samples -- no sliding window. With E
in place, P.E^-1.T_gt^-1 is the SLAM<->gz frame offset D(t), which moves only
with VIO drift, so per-frame noise and re-anchoring merely bend the goal
through the rate limiter below; a window would only blend poses that belong
to different anchors across teleports and turns. The odometry stream itself
is left untouched.

map_node substitute: the gz world plays the role of the built map, and this
node publishes the same contract map_node does -- the TF world->map (C(t))
plus /map/relocalization (the camera pose in gz world, here exact from
ground truth) -- so map-frame consumers (planning's global route, the app
backend, editors) work unchanged against the sim.

The transformed goal is rate-limited and republished on /control/target_pose.
Readiness is latched on /sim/gt_reloc/status (Bool) once both streams have
been seen fresh; scene_runner waits for it before touching the robot.

The gz->SLAM transform only exists in the sim: on a real robot there is no
ground truth, this node does not run, and map_node remains the localization
authority (if /map/relocalization appears, goals pass through untouched).
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

from tinynav.core.math_utils import msg2np, np2msg, np2tf, tf2np
from tf2_ros import TransformBroadcaster

GT_STALE_S = 0.5            # ground truth older than this is not used
P_STALE_S = 1.0             # odometry older than this is not used
MAX_GOAL_SPEED = 1.5        # m/s cap on goal translation updates
MAX_GOAL_YAW_RATE = 1.5     # rad/s cap on goal heading updates

# Static camera extrinsic: lekiwi model origin -> perception camera frame.
# Translation: factory_scene.sdf puts the chassis 0.083 above the model
# origin and infra1_link at (0.09, 0.0255, 0.017) in the chassis frame.
# Rotation: camera axes (x right, y down, z fwd) expressed in the chassis
# frame (x fwd, y left, z up): right -> -y, down -> -z, fwd -> +x. This is
# the transpose of the intuitive "chassis axes in camera coords" matrix --
# as a pose rotation it maps camera-frame points into chassis frame.
E = np.eye(4)
E[:3, :3] = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])
E[:3, 3] = (0.09, 0.0255, 0.100)
E_INV = np.linalg.inv(E)


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


def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


class SimGtGoalRelocator(Node):
    def __init__(self, world, robot_name):
        super().__init__("sim_gt_reloc")
        self.robot_name = robot_name
        self.gt_T = None
        self.gt_wall = 0.0
        self.P = None          # latest perception raw odometry, world->camera 4x4
        self.P_wall = 0.0
        self.goal_gz = None    # latest gz-frame goal (Odometry)
        self.last_out = None   # (monotonic, x, y, yaw) of the last published goal
        self.ready = False

        self.create_subscription(
            TFMessage, f"/world/{world}/pose/info", self.gt_cb,
            QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Odometry, "/slam/odometry_visual_raw", self.raw_cb, 50)
        self.create_subscription(Odometry, "/sim/target_pose_gz", self.goal_gz_cb, 10)
        self.pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.reloc_pub = self.create_publisher(Odometry, "/map/relocalization", 5)
        self.status_pub = self.create_publisher(
            Bool, "/sim/gt_reloc/status",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_timer(0.1, self.tick)
        self.create_timer(1.0, self.reloc_tick)
        self.create_timer(2.0, self.diag_tick)
        self.publish_status(False)
        self.get_logger().info(
            f"sim_gt_reloc up (world={world}, robot={robot_name}); gz goals on "
            "/sim/target_pose_gz -> SLAM frame on /control/target_pose")

    # ---- subscriptions ----

    def gt_cb(self, msg):
        now = time.monotonic()
        for t in msg.transforms:
            if t.child_frame_id.split("/")[-1] != self.robot_name:
                continue
            _, _, T = tf2np(t)
            self.gt_T = T
            self.gt_wall = now

    def raw_cb(self, msg):
        self.P, _ = msg2np(msg)
        self.P_wall = time.monotonic()

    def goal_gz_cb(self, msg):
        self.goal_gz = msg

    # ---- goal forwarding ----

    def current_C(self):
        return self.P @ E_INV @ np.linalg.inv(self.gt_T)

    def streams_ok(self):
        now = time.monotonic()
        return (self.P is not None and now - self.P_wall < P_STALE_S
                and self.gt_T is not None and now - self.gt_wall < GT_STALE_S)

    def diag_tick(self):
        now = time.monotonic()
        p_age = f"{now - self.P_wall:.2f}" if self.P is not None else "never"
        c = "-"
        if self.P is not None and self.gt_T is not None:
            C = self.current_C()
            c = f"({C[0, 3]:.3f},{C[1, 3]:.3f},{C[2, 3]:.3f}) yaw={yaw_of(C):.3f}"
        self.get_logger().info(
            f"diag: P_age={p_age}s gt_age={now - self.gt_wall:.2f}s C={c} "
            f"goal_gz={'yes' if self.goal_gz is not None else 'no'} ready={self.ready}")

    def reloc_tick(self):
        # the "relocalization" a map_node would produce, here exact: the camera
        # pose in the gz (map) world. map_node's convention labels the pose
        # frame "world" -- follow it so consumers parse it identically.
        if self.gt_T is None or time.monotonic() - self.gt_wall > GT_STALE_S:
            return
        cam_in_gz = self.gt_T @ E
        self.reloc_pub.publish(np2msg(cam_in_gz, self.get_clock().now().to_msg(),
                                      "world", "camera"))

    def tick(self):
        # readiness must not depend on a goal having arrived: scene_runner
        # waits for the status latch before publishing anything
        if self.streams_ok() and not self.ready:
            self.ready = True
            self.publish_status(True)
            self.get_logger().info("both streams live; forwarding goals in SLAM frame")
        if not self.streams_ok():
            return
        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        # world->map, gz world standing in for the built map: same contract as
        # map_node's broadcast (planning and the app backend look this up)
        self.tf_broadcaster.sendTransform(
            np2tf(self.current_C(), stamp, "world", "map"))
        if self.goal_gz is None:
            return
        g = self.goal_gz
        C = self.current_C()
        g_slam = C @ np.array([g.pose.pose.position.x,
                               g.pose.pose.position.y,
                               g.pose.pose.position.z, 1.0])
        goal_yaw = yaw_of(C)  # hold-target orientation carries no heading; keep frame yaw
        x, y = g_slam[0], g_slam[1]
        if self.last_out is not None:
            dt = max(now - self.last_out[0], 1e-3)
            dx = np.array([x - self.last_out[1], y - self.last_out[2]])
            step = float(np.linalg.norm(dx))
            cap = MAX_GOAL_SPEED * dt
            if step > cap:
                x, y = (self.last_out[1], self.last_out[2]) + dx / step * cap
            dyaw = (goal_yaw - self.last_out[3] + np.pi) % (2 * np.pi) - np.pi
            goal_yaw = self.last_out[3] + float(np.clip(dyaw, -MAX_GOAL_YAW_RATE * dt,
                                                        MAX_GOAL_YAW_RATE * dt))
        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "world"
        out.child_frame_id = "camera"
        out.pose.pose.position.x = float(x)
        out.pose.pose.position.y = float(y)
        out.pose.pose.position.z = float(g_slam[2])
        out.pose.pose.orientation.w = 1.0
        out.twist = g.twist
        self.pub.publish(out)
        self.last_out = (now, float(x), float(y), float(goal_yaw))

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
    node = SimGtGoalRelocator(world, args.robot_name)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

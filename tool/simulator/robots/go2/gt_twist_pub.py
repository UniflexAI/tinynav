#!/usr/bin/env python3
"""Publish gz ground-truth body twist on /robot1/gt_twist (50 Hz).

Feeds cmd_vel_servo. Parses the /world/<TINYNAV_WORLD_NAME>/pose/info text
stream and finite-differences the model pose into body-frame vx/vy and yaw
rate. TINYNAV_WORLD_NAME must match the world the rig runs (set by
scripts/run_simulator.sh).
"""
import math
import re
import subprocess
import threading
import time

import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

WORLD_NAME = os.environ.get("TINYNAV_WORLD_NAME", "empty")


class GtTwistPub(Node):
    def __init__(self):
        super().__init__("gt_twist_pub")
        self.pub = self.create_publisher(Twist, "gt_twist", 10)
        self.prev = None  # (t, x, y, yaw)
        threading.Thread(target=self.reader, daemon=True).start()

    def on_pose(self, t, x, y, yaw):
        if self.prev is None:
            self.prev = (t, x, y, yaw)
            return
        pt, px, py, pyaw = self.prev
        dt = t - pt
        if 0.02 <= dt <= 0.5:
            vx_w = (x - px) / dt
            vy_w = (y - py) / dt
            dyaw = yaw - pyaw
            while dyaw > math.pi:
                dyaw -= 2 * math.pi
            while dyaw < -math.pi:
                dyaw += 2 * math.pi
            msg = Twist()
            msg.linear.x = math.cos(pyaw) * vx_w + math.sin(pyaw) * vy_w
            msg.linear.y = -math.sin(pyaw) * vx_w + math.cos(pyaw) * vy_w
            msg.angular.z = dyaw / dt
            self.pub.publish(msg)
        self.prev = (t, x, y, yaw)

    def reader(self):
        name_re = re.compile(r'name: "([^"]+)"')
        fvec = re.compile(r'^\s*([xyzw]): ([-+e\d.]+)$')
        while True:
            p = subprocess.Popen(
                ["ign", "topic", "-e", "-t", f"/world/{WORLD_NAME}/pose/info"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            depth = 0
            active = False
            vals = {}
            name = None
            mode = None
            try:
                for line in p.stdout:
                    ls = line.strip()
                    if not active:
                        if ls == "pose {":
                            active, depth, vals, name, mode = True, 1, {}, None, None
                        continue
                    if "{" in ls:
                        depth += ls.count("{")
                        if ls.startswith("position"):
                            mode = "pos"
                        elif ls.startswith("orientation"):
                            mode = "quat"
                        continue
                    if "}" in ls:
                        depth -= ls.count("}")
                        if depth == 0:
                            active = False
                            if name == "go2" and "qw" in vals and "x" in vals:
                                yaw = math.atan2(
                                    2 * (vals["qw"] * vals["qz"] + vals["qx"] * vals["qy"]),
                                    1 - 2 * (vals["qy"] ** 2 + vals["qz"] ** 2))
                                self.on_pose(time.time(), vals["x"], vals["y"], yaw)
                            name, mode, vals = None, None, {}
                        continue
                    m = name_re.match(ls)
                    if m:
                        name = m.group(1)
                        continue
                    if mode:
                        m = fvec.match(ls)
                        if m:
                            vals[("q" if mode == "quat" else "") + m.group(1)] = float(m.group(2))
            except Exception as e:  # stream died; reconnect
                self.get_logger().warn(f"pose stream lost ({e}), reconnecting")
                time.sleep(1.0)
            finally:
                p.kill()


def main():
    rclpy.init()
    node = GtTwistPub()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

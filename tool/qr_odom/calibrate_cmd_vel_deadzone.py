#!/usr/bin/env python3
"""
tool/qr_odom/calibrate_cmd_vel_deadzone.py

One-shot diagnostic: step-sweep a range of constant angular.z commands on
/control/cmd_vel and measure the robot's actual yaw rate from /slam/odometry,
to find the command deadzone (smallest command that actually produces
rotation). Used to sanity-check / retune MIN_ANGULAR in nav_node.py and
mpc_nav_node.py.

Procedure
---------
  For each value in TEST_VALUES (ascending):
    1. idle for IDLE_DURATION_S (command 0, let any prior motion settle)
    2. hold the test value for HOLD_DURATION_S, accumulating measured yaw
       from /slam/odometry (unwrapped, so long holds don't alias)
    3. measured angular rate = (yaw_end - yaw_start) / actual_elapsed
  Prints a (commanded, measured, ratio) table at the end and flags any
  command whose measured/commanded ratio looks deadzone-limited.

Topics
------
  Subscribed:  /slam/odometry     nav_msgs/Odometry
  Published:   /control/cmd_vel   geometry_msgs/Twist

Usage
-----
  python tool/qr_odom/calibrate_cmd_vel_deadzone.py
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from tinynav.core.math_utils import msg2np

ODOM_TOPIC    = "/slam/odometry"
CMD_VEL_TOPIC = "/control/cmd_vel"

# Ascending sweep, fine-grained near the suspected deadzone.
TEST_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
HOLD_DURATION_S = 2.5
IDLE_DURATION_S = 1.5
TICK_HZ = 20.0
DEADZONE_RATIO_THRESH = 0.5   # measured/commanded below this looks deadzone-limited


def _wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class DeadzoneCalibrationNode(Node):
    def __init__(self):
        super().__init__("qr_calibrate_cmd_vel_deadzone")

        self._theta: float | None = None
        self._theta_unwrapped = 0.0
        self._have_odom = False

        self._cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self._odom_cb, 100)

        self._results: list[tuple[float, float, float]] = []
        self._test_idx = -1
        self._phase = "wait_odom"   # wait_odom -> idle -> hold -> idle -> ... -> done
        self._phase_start_time: float | None = None
        self._hold_theta_start = 0.0
        self._done = False

        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(
            f"Deadzone calibration: sweeping angular.z in {TEST_VALUES} "
            f"({HOLD_DURATION_S}s hold / {IDLE_DURATION_S}s idle each). "
            "Waiting for /slam/odometry ..."
        )

    def _odom_cb(self, msg: Odometry) -> None:
        T, _ = msg2np(msg)
        theta = float(np.arctan2(T[1, 0], T[0, 0]))
        if self._theta is not None:
            self._theta_unwrapped += float(_wrap(theta - self._theta))
        self._theta = theta
        self._have_odom = True

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, angular_z: float) -> None:
        cmd = Twist()
        cmd.angular.z = angular_z
        self._cmd_pub.publish(cmd)

    def _tick(self) -> None:
        if self._done:
            return
        now = self._now()

        if self._phase == "wait_odom":
            if not self._have_odom:
                return
            self._phase = "idle"
            self._phase_start_time = now
            self._publish(0.0)
            return

        if self._phase == "idle":
            self._publish(0.0)
            if now - self._phase_start_time >= IDLE_DURATION_S:
                self._test_idx += 1
                if self._test_idx >= len(TEST_VALUES):
                    self._finish()
                    return
                self._phase = "hold"
                self._phase_start_time = now
                self._hold_theta_start = self._theta_unwrapped
            return

        if self._phase == "hold":
            v = TEST_VALUES[self._test_idx]
            self._publish(v)
            elapsed = now - self._phase_start_time
            if elapsed >= HOLD_DURATION_S:
                measured = (self._theta_unwrapped - self._hold_theta_start) / elapsed
                ratio = measured / v if abs(v) > 1e-9 else 0.0
                self._results.append((v, measured, ratio))
                self.get_logger().info(
                    f"commanded={v:+.3f} rad/s -> measured={measured:+.3f} rad/s "
                    f"(ratio={ratio:.2f})"
                )
                self._phase = "idle"
                self._phase_start_time = now
            return

    def _finish(self) -> None:
        self._done = True
        self._publish(0.0)
        self.get_logger().info("Deadzone calibration done. Summary:")
        self.get_logger().info(f"{'commanded':>10}  {'measured':>10}  {'ratio':>6}")
        for v, measured, ratio in self._results:
            self.get_logger().info(f"{v:>10.3f}  {measured:>10.3f}  {ratio:>6.2f}")
        deadzone_candidates = [v for v, _, ratio in self._results if ratio < DEADZONE_RATIO_THRESH]
        if deadzone_candidates:
            self.get_logger().info(
                f"Commands with measured/commanded ratio < {DEADZONE_RATIO_THRESH} "
                f"(likely inside the deadzone): {deadzone_candidates} "
                "-- set MIN_ANGULAR above the largest of these."
            )
        else:
            self.get_logger().info(
                "No tested value looked deadzone-limited (all ratios >= "
                f"{DEADZONE_RATIO_THRESH}) -- try adding smaller test values."
            )


def main(args=None):
    rclpy.init(args=args)
    node = DeadzoneCalibrationNode()
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

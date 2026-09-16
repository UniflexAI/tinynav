"""Upstream's heading control: a lookahead bearing, P-only, and no state.

This fork had replaced it with a PI on the drift between the robot's heading and the
heading of the path pose NEAREST the robot. That reference has no leverage -- the planner
replans from the current pose every cycle, so the nearest pose sits on top of the robot
with the robot's heading -- while the subtraction was large enough to flatten real turns:
measured on go2w, on near-straight stretches the plan asked a median +0.107 rad/s and
what reached /cmd_vel was a median of exactly 0.000.

Upstream instead takes the bearing to a pose `lookahead_steps` ahead, expressed in the
robot frame, and uses it only to REPLACE the command when it is large. These tests pin
the two properties that distinguish that from what was here: the reference moves the
output (leverage), and nothing accumulates between cycles (no integrator).

Frame: with identity pose orientations, cmd_vel_control's T_robot_to_camera makes world
+z the robot's forward axis and world -x its left, so heading_err = atan2(-dx, dz).

Needs rclpy, so this runs in the device container.
"""
from __future__ import annotations

import math
import time
import unittest

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path

from tinynav.platforms.cmd_vel_control import CmdVelControlNode

_DT = 1.0 / 12.0


def _pose(x, y, z):
    p = PoseStamped()
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.position.z = float(z)
    p.pose.orientation.w = 1.0
    return p


def _path(*pts):
    m = Path()
    m.poses = [_pose(*p) for p in pts]
    return m


class _NodeCase(unittest.TestCase):
    """A live CmdVelControlNode with its publisher captured and nav running."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = CmdVelControlNode()
        self.sent: list[Twist] = []
        self.node.cmd_pub.publish = self.sent.append
        self.node._nav_active = True
        self.node._paused = False
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        self.node.pose = odom

    def tearDown(self):
        self.node.destroy_node()

    def _tick(self, ticks=1, before=None):
        """Tick the timer `ticks` times, never stale. Returns the published Twists."""
        n = self.node
        self.sent.clear()
        for _ in range(ticks):
            if before is not None:
                before(n)
            now = time.monotonic()
            n.last_path_update_time = now
            n.last_cmd_pub_time = now - _DT
            n.cmd_timer_callback()
        return list(self.sent)

    def _drive(self, path, ticks=1):
        """Feed one path, then tick. Returns the published Twists."""
        self.node.path_callback(path)
        return self._tick(ticks)


class HeadingControlTest(_NodeCase):
    def test_a_lateral_lookahead_moves_the_output(self):
        """The property the old nearest-pose reference did not have: an offset ahead of
        the robot produces a turn. Straight ahead produces none."""
        straight = self._drive(_path((0, 0, 0), (0, 0, 1.0)))[-1]
        offset = self._drive(_path((0, 0, 0), (-0.5, 0, 1.0)))[-1]
        self.assertAlmostEqual(straight.angular.z, 0.0, places=6)
        self.assertGreater(abs(offset.angular.z), abs(straight.angular.z))

    def test_the_turn_follows_the_side_the_lookahead_is_on(self):
        left = self._drive(_path((0, 0, 0), (-0.5, 0, 1.0)))[-1].angular.z
        right = self._drive(_path((0, 0, 0), (+0.5, 0, 1.0)))[-1].angular.z
        self.assertNotAlmostEqual(left, 0.0, places=3)
        self.assertAlmostEqual(left, -right, places=6)

    def test_a_bigger_bearing_asks_for_a_bigger_turn(self):
        # Monotone up to the clip -- the P in P-only.
        small = abs(self._drive(_path((0, 0, 0), (-0.25, 0, 1.0)))[-1].angular.z)
        large = abs(self._drive(_path((0, 0, 0), (-0.6, 0, 1.0)))[-1].angular.z)
        self.assertGreaterEqual(large, small)

    def test_a_bearing_past_the_force_turn_threshold_stops_the_robot(self):
        # dz small, dx large -> bearing near 90deg, past force_turn_heading_threshold.
        out = self._drive(_path((0, 0, 0), (-1.0, 0, 0.05)))[-1]
        self.assertEqual(out.linear.x, 0.0)
        self.assertGreater(abs(out.angular.z), 0.0)

    def test_nothing_accumulates_between_cycles(self):
        """The anti-integrator property. A PI here would drift the output while the input
        is held fixed; that is exactly how the removed one parked a standing offset and
        how it banked relocalization steps into the yaw command."""
        path = _path((0, 0, 0), (-0.3, 0, 1.0))
        out = [t.angular.z for t in self._drive(path, ticks=200)]
        self.assertEqual(len(set(round(v, 9) for v in out)), 1)

    def test_no_integral_state_survives_on_the_node(self):
        # Guards the reintroduction of the removed PI by name, so a future edit that adds
        # one back trips here rather than on a rig.
        for attr in ('_yaw_bias_per_m', '_drift_lp', 'yaw_kp', 'yaw_bias_ki'):
            self.assertFalse(hasattr(self.node, attr), f'{attr} is back')


class DeadbandTest(_NodeCase):
    """Inside the deadband the output is either a ramp engaging or a stop decaying, and
    the planner's own REQUEST says which -- judged against the engage threshold, not
    against the minimum the chassis can execute. Using the minimum for both made every
    request inside the band read as a stop.
    """

    def setUp(self):
        super().setUp()
        # Straight ahead, so heading control asks for no turn of its own.
        self.node.path_callback(_path((0, 0, 0), (0, 0, 2.0)))

    def _request(self, vx, ticks=40):
        """Ask for `vx` and let the ramp settle. Returns the last published linear.x."""
        def set_cmd(n):
            n.latest_cmd.linear.x = float(vx)
        return self._tick(ticks, before=set_cmd)[-1].linear.x

    def test_a_request_the_chassis_cannot_execute_still_drives(self):
        """Between the engage threshold and the minimum: the planner means to move, so
        the robot moves at the slowest speed it can. This is the case that was zeroed."""
        n = self.node
        self.assertLess(n.linear_engage_threshold, n.min_effective_linear_speed,
                        'the premise is gone: there is no band between the two')
        mid = (n.linear_engage_threshold + n.min_effective_linear_speed) / 2.0
        self.assertAlmostEqual(self._request(mid), n.min_effective_linear_speed,
                               places=6)

    def test_a_request_below_the_engage_threshold_stops(self):
        """The other half, and why this is not just deleting the deadband: the tail of
        a stop must not become a permanent creep."""
        n = self.node
        self.assertEqual(self._request(n.linear_engage_threshold / 2.0), 0.0)

    def test_a_request_above_the_minimum_is_untouched(self):
        n = self.node
        want = n.min_effective_linear_speed * 2.0
        self.assertAlmostEqual(self._request(want), want, places=6)

    def test_raising_the_minimum_does_not_create_a_dead_band(self):
        """Raising the minimum must widen the snap, not the silence: what stops is still
        only what the planner asked to be below the engage threshold."""
        n = self.node
        n.min_effective_linear_speed = 0.2
        self.assertAlmostEqual(self._request(0.15), 0.2, places=6)
        self.assertAlmostEqual(self._request(0.05), 0.2, places=6)
        self.assertEqual(self._request(0.02), 0.0)

    def test_both_axes_go_through_the_same_rule(self):
        """The two halves are one mechanism with different thresholds. Re-inlining either
        one (or giving the turn axis back the minimum-as-discriminator) trips here."""
        snap = self.node._snap
        for min_exec, engage in ((0.2, 0.04), (0.3, 0.3)):
            # Above the minimum: untouched, either sign.
            self.assertEqual(snap(0.5, 0.5, min_exec, engage), 0.5)
            self.assertEqual(snap(-0.5, -0.5, min_exec, engage), -0.5)
            # Inside the band, request still engaging: snap out to the minimum, signed.
            self.assertAlmostEqual(snap(0.01, engage, min_exec, engage), min_exec)
            self.assertAlmostEqual(snap(-0.01, -engage, min_exec, engage), -min_exec)
            # Inside the band, request below the engage threshold: this is a stop.
            self.assertEqual(snap(0.01, engage * 0.9, min_exec, engage), 0.0)


if __name__ == '__main__':
    unittest.main()

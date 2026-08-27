"""Yaw is the planner's feedforward, and nothing in this node modifies it.

A PI on heading drift used to sit between the two. Its reference was the heading of the
published path pose NEAREST the robot, and the planner replans from the current pose
every cycle -- so that pose sits essentially on top of the robot, with essentially the
robot's heading, and the "drift" was near zero by construction. Its subtraction was not
near zero though: measured on go2w over 594 forward-driving samples, the plan asked a
median +0.107 rad/s on near-straight stretches while what went out was a median of
exactly 0.000, and the base drove visibly off to one side. It drove straighter without it.

Needs rclpy, so this runs in the device container.
"""
from __future__ import annotations

import time
import unittest

import rclpy
from geometry_msgs.msg import Twist

from tinynav.platforms.cmd_vel_control import CmdVelControlNode

_DT = 1.0 / 12.0
_POSE_TOPICS = ('/slam/odometry', '/slam/odometry_visual')


class FeedforwardTest(unittest.TestCase):
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

    def tearDown(self):
        self.node.destroy_node()

    def _run(self, omega, vx=0.8, ticks=60):
        n = self.node
        n.latest_cmd.linear.x = vx
        n.path_vyaw_ff = omega
        n.path_vyaw_raw = omega
        n.is_backward_segment = False
        n._forward_speed_cap = 1.5
        n._forward_speed_cap_time = time.monotonic()
        out = []
        for _ in range(ticks):
            now = time.monotonic()
            n.last_path_update_time = now      # never stale
            n.last_cmd_pub_time = now - _DT
            n.cmd_timer_callback()
            out.append(self.sent[-1].angular.z)
        return out

    def test_the_published_yaw_is_the_feedforward(self):
        floor = self.node.min_effective_angular_speed
        for omega in (floor, 0.2, 0.4, -0.3, -0.6):
            for z in self._run(omega):
                self.assertAlmostEqual(z, omega, places=6, msg=f'omega={omega}')

    def test_it_is_clipped_to_what_the_base_can_turn(self):
        # Driven through the real entry point, because that is where the clip lives and
        # it also scales vx to preserve the turn radius. The omega has to sit between
        # max_angular_speed and rotate_first_omega: above the latter, rotate-first owns
        # the command instead (asserted below), so a larger value tests that path, not
        # this one.
        n = self.node
        cap = n.max_angular_speed
        omega = (cap + n.rotate_first_omega) / 2.0
        self.assertGreater(omega, cap)
        self.assertLess(omega, n.rotate_first_omega)
        for sign in (+1.0, -1.0):
            ff = Twist()
            ff.linear.x, ff.angular.z = 0.8, sign * omega
            n.velocity_ff_callback(ff)
            self.assertAlmostEqual(n.path_vyaw_ff, sign * cap, places=6)
            now = time.monotonic()
            n.last_path_update_time = now
            n.last_cmd_pub_time = now - _DT
            n.cmd_timer_callback()
            self.assertAlmostEqual(self.sent[-1].angular.z, sign * cap, places=6)

    def test_a_turn_too_sharp_to_drive_through_becomes_a_turn_in_place(self):
        # Not the clip: rotate-first replaces the command outright, and it is keyed off
        # the UNCLAMPED omega so the clip above cannot hide it.
        n = self.node
        ff = Twist()
        ff.linear.x, ff.angular.z = 0.8, n.rotate_first_omega * 1.5
        n.velocity_ff_callback(ff)
        now = time.monotonic()
        n.last_path_update_time = now
        n.last_cmd_pub_time = now - _DT
        n.cmd_timer_callback()
        self.assertAlmostEqual(self.sent[-1].angular.z, n.rotate_first_max_omega,
                               places=6)
        self.assertEqual(self.sent[-1].linear.x, 0.0)

    def test_a_constant_feedforward_produces_a_constant_command(self):
        # The failure mode a feedback term reintroduces is drift over time: same input,
        # changing output. Sub-floor omegas are excluded -- those are deliberately
        # duty-cycled (test_cmd_vel_yaw_bank.py).
        z = self._run(0.3, ticks=200)
        self.assertEqual(len(set(round(v, 9) for v in z)), 1)

    def test_there_is_no_pose_subscription_left(self):
        # The heading reference is gone, so the node must not be reading odometry at all;
        # a subscription still here would mean a feedback path came back.
        topics = {s.topic_name for s in self.node.subscriptions}
        for t in _POSE_TOPICS:
            self.assertNotIn(t, topics)

    def test_it_still_subscribes_to_what_it_does_use(self):
        # Guards the assertion above against passing because the node stopped
        # subscribing to anything at all.
        topics = {s.topic_name for s in self.node.subscriptions}
        for t in ('/planning/velocity_ff', '/planning/trajectory_path',
                  '/planning/forward_speed_cap', '/nav/active', '/nav/paused'):
            self.assertIn(t, topics)


if __name__ == '__main__':
    unittest.main()

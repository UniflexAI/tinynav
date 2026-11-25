"""The brake on a nav release.

`/nav/active` going false is where nav hands `/cmd_vel` back. It used to publish one
all-zero Twist there, which is a step, not a ramp: measured on poi5 2026-11-24 the
arrival fired while the planner was still commanding 0.60 m/s with 1.10 m of path
left, so the chassis got `0.60 -> 0` in one message and read as slamming on. The
stick has its own exponential brake (pilot's teleop.py); this is the same rate for
the same feel.

These pin the three properties that make it a brake: it sheds instead of stepping,
it ends on exactly one definite zero and then stays silent, and it yields to both
things that can take the bus back (nav re-engaging, the operator pausing).

Needs rclpy, so this runs in the device container:
    python3 -m unittest tests.test_cmd_vel_brake
"""
from __future__ import annotations

import math
import unittest

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from tests.test_cmd_vel_heading import _DT, _NodeCase


class NavReleaseBrakeTest(_NodeCase):
    def _last_command(self, vx: float, wz: float = 0.0) -> None:
        """What `/cmd_vel` was last told, so the release has something to shed."""
        self.node.prev_cmd = Twist()
        self.node.prev_cmd.linear.x = vx
        self.node.prev_cmd.angular.z = wz

    def test_a_release_sheds_instead_of_stepping_to_zero(self):
        self._last_command(0.6)
        self.node._on_nav_active(Bool(data=False))
        out = self._tick()
        self.assertEqual(len(out), 1)
        # One exponential step at the stick's rate, not the previous value and
        # not zero: 0.6 * exp(-6 * 1/12). The tick's own dt is the measured one,
        # so the comparison carries a tolerance.
        self.assertAlmostEqual(out[0].linear.x, 0.6 * math.exp(-6.0 * _DT), delta=1e-3)

    def test_the_brake_ends_on_one_exact_zero_and_then_goes_quiet(self):
        self._last_command(0.6, wz=0.3)
        self.node._on_nav_active(Bool(data=False))
        shed = []
        for _ in range(40):
            out = self._tick()
            if not out:
                break
            shed.append(out[0].linear.x)
        # It ramped (several steps), monotonically, and the last one is the zero.
        self.assertGreaterEqual(len(shed), 3)
        self.assertEqual(shed, sorted(shed, reverse=True))
        self.assertEqual(shed[-1], 0.0)
        self.assertTrue(all(v > 0.0 for v in shed[:-1]))
        # Silent afterwards: teleop owns the bus again.
        self.assertEqual(self._tick(), [])

    def test_the_turn_is_shed_too(self):
        self._last_command(0.0, wz=0.3)
        self.node._on_nav_active(Bool(data=False))
        out = self._tick()
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].angular.z, 0.3 * math.exp(-6.0 * _DT), delta=1e-3)

    def test_nav_re_engaging_cancels_the_brake(self):
        self._last_command(0.6)
        self.node._on_nav_active(Bool(data=False))
        self._tick()
        self.assertIsNotNone(self.node._braking)
        self.node._on_nav_active(Bool(data=True))
        self.assertIsNone(self.node._braking)

    def test_a_pause_ends_the_brake_on_a_zero(self):
        self._last_command(0.6)
        self.node._on_nav_active(Bool(data=False))
        self._tick()
        self.sent.clear()
        self.node._on_paused(Bool(data=True))
        self.assertIsNone(self.node._braking)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0].linear.x, 0.0)

    def test_a_release_from_a_standstill_is_still_one_zero(self):
        self._last_command(0.0)
        self.node._on_nav_active(Bool(data=False))
        out = self._tick()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].linear.x, 0.0)
        self.assertEqual(out[0].angular.z, 0.0)


if __name__ == '__main__':
    unittest.main()

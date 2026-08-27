"""A turn request below the chassis's executable floor must still be executed.

It used to be published as exactly 0 while linear.x kept its accel-limited value, so
the robot drove straight through gentle corners: measured on go2w, 43% of the commands
issued while the planner was asking for a turn went out as 0.00 rad/s at a mean
0.53 m/s. A heading-only controller cannot recover that -- the robot was commanded off
the path, not pushed off it.

Needs rclpy, so this runs in the container (see the repo's test notes).
"""
from __future__ import annotations

import time
import unittest

import rclpy
from geometry_msgs.msg import Twist

from tinynav.platforms.cmd_vel_control import CmdVelControlNode

_DT = 1.0 / 12.0        # the node's own cmd_rate_hz
_TICKS = 240


class YawBankTest(unittest.TestCase):
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

    def _run(self, omega, vx=0.8, ticks=_TICKS):
        """Drive the timer callback `ticks` times with a fixed dt and a fresh path.

        pose/path are left None so _path_intended_yaw returns None and the drift PI
        contributes nothing -- this isolates the floor logic from the feedback term.
        """
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
            n.last_path_update_time = now          # never stale
            n.last_cmd_pub_time = now - _DT        # forces dt == _DT
            n.cmd_timer_callback()
            out.append(self.sent[-1].angular.z)
        return out

    def test_every_published_turn_rate_is_executable(self):
        floor = self.node.min_effective_angular_speed
        for omega in (0.01, 0.03, 0.05, 0.09):
            for z in self._run(omega):
                self.assertTrue(z == 0.0 or abs(z) >= floor - 1e-9,
                                f'omega={omega} published un-executable {z}')

    def test_the_mean_turn_rate_is_the_one_requested(self):
        # The property that makes banking correct: the chassis only ever sees the floor
        # or zero, but the time-average over a few ticks is the request.
        for omega in (0.02, 0.04, 0.06, 0.08):
            z = self._run(omega)
            self.assertAlmostEqual(sum(z) / len(z), omega, delta=0.01,
                                   msg=f'omega={omega}')

    def test_a_sub_floor_request_is_not_dropped(self):
        # The regression itself: this used to be all zeros.
        for omega in (0.02, 0.05, 0.09):
            z = self._run(omega)
            self.assertGreater(sum(1 for v in z if v != 0.0), 0, f'omega={omega}')

    def test_the_sign_is_never_inverted(self):
        for omega in (0.04, -0.04):
            for z in self._run(omega):
                self.assertFalse(z * omega < 0.0, f'omega={omega} turned the wrong way')

    def test_zero_request_stays_zero(self):
        # Banking must not invent a turn where the plan asked for none.
        self.assertEqual(set(self._run(0.0)), {0.0})

    def test_requests_above_the_floor_pass_through_untouched(self):
        floor = self.node.min_effective_angular_speed
        for omega in (floor, 0.3, 0.6):
            z = self._run(omega, ticks=20)
            for v in z:
                self.assertAlmostEqual(v, omega, places=6, msg=f'omega={omega}')

    def test_the_bank_does_not_survive_a_nav_stop(self):
        # Yaw banked before a stop must not be spent after the next start, where the
        # plan is a different one.
        self._run(0.09, ticks=5)
        self.node._on_nav_active(_bool(False))
        self.assertEqual(self.node._yaw_bank, 0.0)

    def test_the_bank_stays_bounded(self):
        self._run(0.09)
        self.assertLessEqual(abs(self.node._yaw_bank),
                             self.node.yaw_bank_limit + 1e-9)


def _bool(v):
    from std_msgs.msg import Bool
    m = Bool()
    m.data = v
    return m


if __name__ == '__main__':
    unittest.main()

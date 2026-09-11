"""unitree_control's rt/cmd_vel path and its chassis watchdog.

ClassicWalk and StopMove are reply RPCs that wait up to the client timeout on the
rt/cmd_vel reader thread; ClassicWalk used to be sent before every Move. These pin
that ClassicWalk is once per motion start, that a stop is a zero Move, and that the
diagnostics say something when the chassis stops executing and nothing when it does.

Needs unitree_sdk2py and rclpy, so this runs in the device container.
"""
from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault('ROBOT_TYPE', 'b2')

from tinynav.platforms import unitree_control as uc  # noqa: E402


class _Log:
    def __init__(self):
        self.lines = []

    def _add(self, level):
        return lambda msg: self.lines.append((level, msg))

    def __getattr__(self, level):
        if level in ('debug', 'info', 'warning', 'error'):
            return self._add(level)
        if level == 'exception':
            return self._add('error')
        raise AttributeError(level)

    def has(self, level, text):
        return any(lv == level and text in m for lv, m in self.lines)


class _Sport:
    def __init__(self, walk_delay=0.0):
        self.calls = []
        self.walk_delay = walk_delay

    def ClassicWalk(self, flag):
        time.sleep(self.walk_delay)
        self.calls.append('ClassicWalk')
        return 0

    def Move(self, vx, vy, wz):
        self.calls.append('Move' if (vx, vy, wz) != (0.0, 0.0, 0.0) else 'Move0')
        return 0

    def StopMove(self):
        self.calls.append('StopMove')
        return 0

    def count(self, name):
        return self.calls.count(name)


def _node(sport=None):
    node = uc.Ros2UnitreeManagerNode.__new__(uc.Ros2UnitreeManagerNode)
    node.logger = _Log()
    node.sport_client = sport or _Sport()
    node.is_quadruped = True
    node.last_twist_time = None
    node._walking = False
    node._gait_due = False
    node._move_failures = 0
    node._move_failure_logged_at = None
    node.watch = uc.ChassisWatch(node.logger)
    return node


def _drive(node, t0, n, v, dt=1.0 / 12.0):
    for i in range(n):
        node._on_twist(t0 + i * dt, v, 0.0, 0.0)
    return t0 + n * dt


class TestTwistPath(unittest.TestCase):
    def test_classic_walk_once_per_motion_start_not_per_move(self):
        node = _node()
        t = _drive(node, 0.0, 10, 0.5)
        t = _drive(node, t, 5, 0.0)
        _drive(node, t, 10, 0.5)
        sport = node.sport_client
        self.assertEqual(sport.count('Move'), 20)
        self.assertEqual(sport.count('ClassicWalk'), 2)
        # Each ClassicWalk precedes the first Move of its motion.
        self.assertEqual(sport.calls[0], 'ClassicWalk')
        last_zero = len(sport.calls) - 1 - sport.calls[::-1].index('Move0')
        self.assertEqual(sport.calls[last_zero + 1], 'ClassicWalk')

    def test_zero_command_is_a_zero_move_never_stopmove(self):
        node = _node()
        t = _drive(node, 0.0, 12, 0.5)
        _drive(node, t, 12, 0.0)
        sport = node.sport_client
        self.assertEqual(sport.count('StopMove'), 0)
        self.assertEqual(sport.count('Move0'), 12)
        self.assertEqual(sport.calls[-12:], ['Move0'] * 12)

    def test_gap_mid_motion_is_logged_and_reasserts_the_gait(self):
        node = _node()
        t = _drive(node, 0.0, 12, 0.5)
        _drive(node, t + 1.0, 3, 0.5)
        self.assertTrue(node.logger.has('warning', 'rt/cmd_vel silent'))
        self.assertEqual(node.sport_client.count('ClassicWalk'), 2)

    def test_steady_stream_logs_no_gap(self):
        node = _node()
        _drive(node, 0.0, 36, 0.5)
        self.assertFalse(node.logger.has('warning', 'silent'))
        self.assertEqual(node.sport_client.count('ClassicWalk'), 1)

    def test_slow_reply_is_logged(self):
        node = _node(_Sport(walk_delay=uc._SLOW_RPC_S + 0.05))
        node._on_twist(0.0, 0.5, 0.0, 0.0)
        self.assertTrue(node.logger.has('warning', '[sport] ClassicWalk'))

    def test_fast_reply_is_not_logged(self):
        node = _node()
        node._on_twist(0.0, 0.5, 0.0, 0.0)
        self.assertFalse(node.logger.has('warning', '[sport]'))

    def test_move_send_failure_is_logged_rate_limited(self):
        sport = _Sport()
        sport.Move = lambda vx, vy, wz: 3102
        node = _node(sport)
        _drive(node, 0.0, 24, 0.5)
        fails = [m for lv, m in node.logger.lines if 'Move send failed' in m]
        self.assertEqual(len(fails), 1)


class TestChassisWatch(unittest.TestCase):
    def _run(self, cmd_v, chassis_v, seconds, watch=None, t0=0.0):
        log = watch.log if watch else _Log()
        watch = watch or uc.ChassisWatch(log)
        t = t0
        while t < t0 + seconds:
            watch.on_cmd(t, cmd_v, 0.0, 0.0)
            watch.on_sport_state(t, 7, 1, 0, (chassis_v, 0.0, 0.0), 0.0, (1.0, 1.0, 1.0, 1.0))
            watch.check(t)
            t += 0.1
        return watch, t

    def test_commanded_but_still_is_logged(self):
        watch, _ = self._run(0.56, 0.0, 3.0)
        self.assertTrue(watch.log.has('warning', 'not executing'))

    def test_commanded_and_moving_is_not_logged(self):
        watch, _ = self._run(0.56, 0.4, 3.0)
        self.assertFalse(watch.log.has('warning', 'not executing'))

    def test_small_command_is_not_a_stall(self):
        watch, _ = self._run(0.05, 0.0, 3.0)
        self.assertFalse(watch.log.has('warning', 'not executing'))

    def test_recovery_is_logged_once_the_chassis_moves(self):
        watch, t = self._run(0.56, 0.0, 3.0)
        self._run(0.56, 0.4, 0.5, watch=watch, t0=t)
        self.assertTrue(watch.log.has('info', 'executing again'))

    def test_mode_change_is_logged(self):
        log = _Log()
        watch = uc.ChassisWatch(log)
        watch.on_sport_state(0.0, 7, 1, 0, (0, 0, 0), 0.0, ())
        watch.on_sport_state(0.1, 7, 1, 0, (0, 0, 0), 0.0, ())
        watch.on_sport_state(0.2, 1, 0, 0, (0, 0, 0), 0.0, ())
        changes = [m for lv, m in log.lines if m.startswith('[chassis] mode=')]
        self.assertEqual(len(changes), 2)
        self.assertIn('was mode=7', changes[-1])

    def test_silent_sport_state_is_logged(self):
        log = _Log()
        watch = uc.ChassisWatch(log)
        watch.on_sport_state(0.0, 7, 1, 0, (0, 0, 0), 0.0, ())
        watch.check(0.5)
        self.assertFalse(log.has('warning', 'silent'))
        watch.check(2.0)
        self.assertTrue(log.has('warning', 'rt/sportmodestate silent'))


if __name__ == '__main__':
    unittest.main()

"""unitree_control's rt/cmd_vel path, its gait worker, and its chassis watchdog.

ClassicWalk and StopMove are reply RPCs that wait up to the client timeout. Run on
the rt/cmd_vel reader thread they hold every queued Move behind them, which is what
the robot's walk-two-steps-and-stop was. These pin that the reader thread never waits
on a reply RPC, that the gait is asserted once per motion start, that a stop is a zero
Move, and that the diagnostics speak up when the chassis stops executing and stay
quiet when it does not.

Needs unitree_sdk2py and rclpy, so this runs in the device container.
"""
from __future__ import annotations

import os
import threading
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
        self.walk_entered = threading.Event()

    def ClassicWalk(self, flag):
        self.walk_entered.set()
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


class _GaitSpy:
    """Stands in for GaitWorker: counts hand-offs without starting a thread."""

    def __init__(self):
        self.requests = 0

    def request(self):
        self.requests += 1


def _node(sport=None, gait=None):
    node = uc.Ros2UnitreeManagerNode.__new__(uc.Ros2UnitreeManagerNode)
    node.logger = _Log()
    node.sport_client = sport or _Sport()
    node.is_quadruped = True
    node.last_twist_time = None
    node._walking = False
    node._gait_due = False
    node._gait_held = False
    node._move_failures = 0
    node._move_failure_logged_at = None
    node.watch = uc.ChassisWatch(node.logger)
    node.gait = gait if gait is not None else _GaitSpy()
    return node


def _drive(node, t0, n, v, dt=1.0 / 12.0):
    for i in range(n):
        node._on_twist(t0 + i * dt, v, 0.0, 0.0)
    return t0 + n * dt


class TestTwistPath(unittest.TestCase):
    def test_gait_requested_once_per_motion_start_not_per_move(self):
        node = _node()
        t = _drive(node, 0.0, 10, 0.5)
        t = _drive(node, t, 5, 0.0)
        _drive(node, t, 10, 0.5)
        self.assertEqual(node.sport_client.count('Move'), 20)
        self.assertEqual(node.gait.requests, 2)

    def test_the_reader_thread_never_calls_the_gait_rpc(self):
        node = _node()
        _drive(node, 0.0, 10, 0.5)
        self.assertEqual(node.sport_client.count('ClassicWalk'), 0)

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
        self.assertEqual(node.gait.requests, 2)

    def test_a_held_gait_is_not_asked_for_again(self):
        # A dock's final approach starts and stops every second; each start used to
        # send ClassicWalk, and one that timed out held the sport service while the
        # dock's sit arrived, so the sit timed out too (65, 2026-09-29).
        node = _node()
        node._note_gait(0)
        t = _drive(node, 0.0, 10, 0.5)
        t = _drive(node, t, 5, 0.0)
        _drive(node, t, 10, 0.5)
        self.assertEqual(node.gait.requests, 0)

    def test_a_failed_assertion_is_asked_for_at_the_next_start(self):
        node = _node()
        node._note_gait(3104)
        t = _drive(node, 0.0, 10, 0.5)
        t = _drive(node, t, 5, 0.0)
        _drive(node, t, 10, 0.5)
        self.assertEqual(node.gait.requests, 2)

    def test_a_stand_is_followed_by_one_assertion(self):
        # A stand's own ClassicWalk returns 0 within 10ms, before the robot is up;
        # counted as held, it left the chassis in BalanceStand ignoring every Move
        # (65, 2026-09-29). The next motion start asserts it once, and only that
        # call's 0 holds it.
        node = _node()
        node._robot_status = None
        node._note_gait(0)
        node._play_steps('Standing', [('StandUp', lambda: 0),
                                      ('ClassicWalk', lambda: 0)], 'standup')
        t = _drive(node, 0.0, 10, 0.5)
        self.assertEqual(node.gait.requests, 1)
        node._note_gait(0)
        t = _drive(node, t, 5, 0.0)
        _drive(node, t, 10, 0.5)
        self.assertEqual(node.gait.requests, 1)

    def test_a_sit_clears_the_gait(self):
        node = _node()
        node._robot_status = None
        node._note_gait(0)
        node._play_steps('Sitting', [('StandDown', lambda: 0)], 'sitting')
        self.assertFalse(node._gait_held)

    def test_the_worker_reports_each_result(self):
        seen = []
        worker = uc.GaitWorker(lambda: 0, _Log(), on_result=seen.append)
        worker.run_once()
        worker = uc.GaitWorker(lambda: (_ for _ in ()).throw(RuntimeError()), _Log(),
                               on_result=seen.append)
        worker.run_once()
        self.assertEqual(seen, [0, None])

    def test_steady_stream_logs_no_gap(self):
        node = _node()
        _drive(node, 0.0, 36, 0.5)
        self.assertFalse(node.logger.has('warning', 'silent'))
        self.assertEqual(node.gait.requests, 1)

    def test_move_send_failure_is_logged_rate_limited(self):
        sport = _Sport()
        sport.Move = lambda vx, vy, wz: 3102
        node = _node(sport)
        _drive(node, 0.0, 24, 0.5)
        fails = [m for lv, m in node.logger.lines if 'Move send failed' in m]
        self.assertEqual(len(fails), 1)


class TestGaitWorker(unittest.TestCase):
    """The reply RPC lives here now, so this is where the 10s wait has to be absorbed."""

    def test_a_stuck_gait_rpc_does_not_hold_up_move(self):
        # The regression this whole change is about: a ClassicWalk that never gets a
        # reply used to sit on the reader thread and starve Move.
        released = threading.Event()
        sport = _Sport()
        sport.ClassicWalk = lambda flag: (sport.walk_entered.set(), released.wait(5), 0)[-1]
        node = _node(sport)
        node.gait = uc.GaitWorker(lambda: sport.ClassicWalk(True), node.logger)
        node.gait.start()
        try:
            t0 = time.monotonic()
            _drive(node, 0.0, 12, 0.5)
            elapsed = time.monotonic() - t0
            self.assertTrue(sport.walk_entered.wait(2), 'the worker never ran the RPC')
            self.assertEqual(sport.count('Move'), 12)
            self.assertLess(elapsed, 1.0)
        finally:
            released.set()
            node.gait.stop()

    def test_calling_the_stuck_rpc_inline_does_block(self):
        # Pairs with the test above: shows the timing assertion can fail.
        blocked = uc.GaitWorker(lambda: time.sleep(0.3) or 0, _Log())
        t0 = time.monotonic()
        blocked.run_once()
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)

    def test_slow_reply_is_logged(self):
        log = _Log()
        uc.GaitWorker(lambda: time.sleep(uc._SLOW_RPC_S + 0.05) or 0, log).run_once()
        self.assertTrue(log.has('warning', '[sport] ClassicWalk'))

    def test_fast_reply_is_not_logged(self):
        log = _Log()
        uc.GaitWorker(lambda: 0, log).run_once()
        self.assertFalse(log.has('warning', '[sport]'))

    def test_nonzero_code_is_logged_however_fast(self):
        log = _Log()
        uc.GaitWorker(lambda: 3104, log).run_once()
        self.assertTrue(log.has('warning', 'code=3104'))

    def test_a_raising_rpc_does_not_kill_the_worker(self):
        log = _Log()
        calls = []

        def call():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('boom')
            return 0

        worker = uc.GaitWorker(call, log)
        worker.start()
        try:
            worker.request()
            deadline = time.monotonic() + 2
            while len(calls) < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(log.has('error', 'raised'))
            worker.request()
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(calls), 2)
        finally:
            worker.stop()

    def test_requests_during_a_call_coalesce_into_one(self):
        released = threading.Event()
        entered = threading.Event()
        calls = []

        def call():
            calls.append(1)
            entered.set()
            released.wait(5)
            return 0

        worker = uc.GaitWorker(call, _Log())
        worker.start()
        try:
            worker.request()
            self.assertTrue(entered.wait(2))
            for _ in range(20):
                worker.request()
            released.set()
            deadline = time.monotonic() + 2
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            # 21 requests, two calls: the first, and one standing for all the rest.
            self.assertEqual(len(calls), 2)
        finally:
            released.set()
            worker.stop()


class TestActionWorker(unittest.TestCase):
    """65, 2026-09-29: a stand queued behind a 20s refused one ran with the robot
    walking at 0.9 m/s. The stamp is taken on arrival, so waiting is measurable."""

    def _worker(self):
        played, log = [], _Log()
        return uc.ActionWorker(played.append, log), played, log

    def test_a_command_run_on_arrival_is_played(self):
        worker, played, _ = self._worker()
        worker.run_once(10.0, 'play sit', 10.1)
        self.assertEqual(played, ['play sit'])

    def test_a_command_that_waited_too_long_is_dropped(self):
        worker, played, log = self._worker()
        worker.run_once(10.0, 'play stand', 10.0 + uc._ACTION_MAX_WAIT_S + 0.1)
        self.assertEqual(played, [])
        self.assertTrue(log.has('warning', 'dropped'))

    def test_one_behind_a_stuck_call_is_dropped_and_the_next_is_not(self):
        """Through the thread: the reader stamps on arrival while the first call
        blocks, and only the one that waited past the limit is lost."""
        played, gate = [], threading.Event()

        def play(data):
            if data == 'play stand' and not played:
                gate.wait(2.0)
            played.append(data)

        worker = uc.ActionWorker(play, _Log())
        worker.start()
        t0 = time.monotonic()
        worker.submit('play stand', t0)
        worker.submit('play stand', t0 - uc._ACTION_MAX_WAIT_S)   # already stale
        gate.set()
        worker.submit('play sit', time.monotonic())
        worker.stop()
        self.assertEqual(played, ['play stand', 'play sit'])

    def test_a_raising_play_does_not_kill_the_worker(self):
        played = []

        def play(data):
            if data == 'bad':
                raise RuntimeError('boom')
            played.append(data)

        log = _Log()
        worker = uc.ActionWorker(play, log)
        worker.start()
        now = time.monotonic()
        worker.submit('bad', now)
        worker.submit('play sit', now)
        worker.stop()
        self.assertEqual(played, ['play sit'])
        self.assertTrue(log.has('error', 'action failed'))

    def test_the_reader_only_queues(self):
        node = _node()
        node.actions = uc.ActionWorker(lambda d: self.fail('played on the reader'), node.logger)
        node.ActionMessageHandler(type('M', (), {'data': 'play sit'})())
        self.assertEqual(node.actions._queue.qsize(), 1)


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

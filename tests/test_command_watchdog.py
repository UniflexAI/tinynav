import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tinynav.platforms.command_watchdog import VelocityCommandWatchdog, stop_unitree_motion


class FakeMotionClient:
    def __init__(self):
        self.calls = []

    def SetVelocity(self, vx, vy, vyaw):
        self.calls.append(('SetVelocity', vx, vy, vyaw))
        return 0

    def StopMove(self):
        self.calls.append(('StopMove',))
        return 0


def test_rejects_invalid_timeout():
    for timeout_s in (0.0, -0.1, float('inf'), float('nan')):
        try:
            VelocityCommandWatchdog(timeout_s)
        except ValueError:
            continue
        raise AssertionError(f'timeout {timeout_s} should fail')


def test_idle_does_not_expire():
    watchdog = VelocityCommandWatchdog(0.5)
    assert not watchdog.consume_expiration(10.0)


def test_nonzero_expires_once():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    assert generation == 1
    assert not watchdog.consume_expiration(10.5)
    assert watchdog.consume_expiration(10.5001)
    assert not watchdog.consume_expiration(20.0)


def test_refresh_extends_deadline():
    watchdog = VelocityCommandWatchdog(0.5)
    assert watchdog.observe_nonzero(10.0) == 1
    assert watchdog.observe_nonzero(10.4) == 2
    assert not watchdog.consume_expiration(10.8)
    assert watchdog.consume_expiration(10.9001)


def test_nonzero_attempt_stays_armed_until_confirmed_stop():
    watchdog = VelocityCommandWatchdog(0.5)
    watchdog.observe_nonzero(10.0)
    assert watchdog.consume_expiration(10.5001)


def test_confirmed_stop_disarms_watchdog():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    stop_generation = watchdog.request_stop()
    assert stop_generation == generation
    watchdog.confirm_stop(stop_generation)
    assert not watchdog.consume_expiration(20.0)


def test_late_completion_requires_compensating_stop():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    assert watchdog.consume_expiration(10.5001)
    assert watchdog.completed_after_stop(generation)


def test_normal_completion_does_not_require_stop():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    assert not watchdog.completed_after_stop(generation)


def test_explicit_stop_covers_inflight_generation():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    watchdog.request_stop()
    assert watchdog.completed_after_stop(generation)
    assert watchdog.consume_expiration(10.5001)


def test_unconfirmed_stop_keeps_watchdog_armed():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    watchdog.request_stop()
    assert watchdog.consume_expiration(10.5001)
    assert watchdog.completed_after_stop(generation)


def test_confirming_old_stop_does_not_disarm_new_generation():
    watchdog = VelocityCommandWatchdog(0.5)
    first_generation = watchdog.observe_nonzero(10.0)
    stop_generation = watchdog.request_stop()
    second_generation = watchdog.observe_nonzero(10.1)
    assert second_generation > first_generation
    watchdog.confirm_stop(stop_generation)
    assert watchdog.consume_expiration(10.6001)


def test_current_unconfirmed_stop_failure_retries():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    assert watchdog.consume_expiration(10.5001) == generation
    assert watchdog.retry_after_stop_failure(generation, 10.6)
    assert watchdog.consume_expiration(11.1001) == generation


def test_old_stop_failure_does_not_delay_new_generation():
    watchdog = VelocityCommandWatchdog(0.5)
    first_generation = watchdog.observe_nonzero(10.0)
    assert watchdog.consume_expiration(10.5001) == first_generation
    second_generation = watchdog.observe_nonzero(10.6)
    assert not watchdog.retry_after_stop_failure(first_generation, 20.0)
    assert watchdog.consume_expiration(11.1001) == second_generation


def test_confirmed_stop_failure_does_not_rearm_idle_watchdog():
    watchdog = VelocityCommandWatchdog(0.5)
    generation = watchdog.observe_nonzero(10.0)
    stop_generation = watchdog.request_stop()
    watchdog.confirm_stop(stop_generation)
    assert not watchdog.retry_after_stop_failure(generation, 20.0)
    assert watchdog.consume_expiration(30.0) is None


def test_stop_uses_status_returning_g1_api():
    client = FakeMotionClient()
    assert stop_unitree_motion(client, 'g1') == 0
    assert client.calls == [('SetVelocity', 0.0, 0.0, 0.0)]


def test_stop_uses_sport_api_for_quadruped():
    client = FakeMotionClient()
    assert stop_unitree_motion(client, 'go2w') == 0
    assert client.calls == [('StopMove',)]


if __name__ == '__main__':
    tests = [
        test_rejects_invalid_timeout,
        test_idle_does_not_expire,
        test_nonzero_expires_once,
        test_refresh_extends_deadline,
        test_nonzero_attempt_stays_armed_until_confirmed_stop,
        test_confirmed_stop_disarms_watchdog,
        test_late_completion_requires_compensating_stop,
        test_normal_completion_does_not_require_stop,
        test_explicit_stop_covers_inflight_generation,
        test_unconfirmed_stop_keeps_watchdog_armed,
        test_confirming_old_stop_does_not_disarm_new_generation,
        test_current_unconfirmed_stop_failure_retries,
        test_old_stop_failure_does_not_delay_new_generation,
        test_confirmed_stop_failure_does_not_rearm_idle_watchdog,
        test_stop_uses_status_returning_g1_api,
        test_stop_uses_sport_api_for_quadruped,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')

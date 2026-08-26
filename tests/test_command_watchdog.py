import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tinynav.platforms.command_watchdog import VelocityCommandWatchdog


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
    watchdog.observe_nonzero(10.0)
    assert not watchdog.consume_expiration(10.5)
    assert watchdog.consume_expiration(10.5001)
    assert not watchdog.consume_expiration(20.0)


def test_refresh_extends_deadline():
    watchdog = VelocityCommandWatchdog(0.5)
    watchdog.observe_nonzero(10.0)
    watchdog.observe_nonzero(10.4)
    assert not watchdog.consume_expiration(10.8)
    assert watchdog.consume_expiration(10.9001)


def test_nonzero_attempt_stays_armed_until_confirmed_stop():
    watchdog = VelocityCommandWatchdog(0.5)
    watchdog.observe_nonzero(10.0)
    assert watchdog.consume_expiration(10.5001)


def test_clear_disarms_watchdog():
    watchdog = VelocityCommandWatchdog(0.5)
    watchdog.observe_nonzero(10.0)
    watchdog.clear()
    assert not watchdog.consume_expiration(20.0)


if __name__ == '__main__':
    tests = [
        test_rejects_invalid_timeout,
        test_idle_does_not_expire,
        test_nonzero_expires_once,
        test_refresh_extends_deadline,
        test_nonzero_attempt_stays_armed_until_confirmed_stop,
        test_clear_disarms_watchdog,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')

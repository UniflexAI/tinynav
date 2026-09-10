import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'platforms'))
from cmd_vel_control import compute_heading_err


def test_heading_err_ignores_near_zero_displacement_noise():
    # regression: a rosbag showed the robot near-stationary (planned segment
    # displacement ~1cm) with atan2(p[1], p[0]) swinging across the full
    # +/-pi range purely from position noise, occasionally exceeding the
    # forced-turn gate and injecting a random-direction rotate command -
    # reported as the robot "walking like it's drunk"
    rng = np.random.default_rng(0)
    for _ in range(200):
        noise = rng.normal(scale=0.005, size=2)  # ~5mm noise, well under min_displacement
        assert compute_heading_err(noise) == 0.0


def test_heading_err_trusts_real_displacement():
    # a genuine forward-and-to-the-side segment should still report its real bearing
    p = np.array([0.5, 0.5, 0.0])
    got = compute_heading_err(p)
    assert abs(got - np.pi / 4) < 1e-9


def test_heading_err_threshold_boundary():
    just_under = np.array([0.02, 0.0, 0.0])
    just_over = np.array([0.05, 0.0, 0.0])
    assert compute_heading_err(just_under, min_displacement=0.03) == 0.0
    assert compute_heading_err(just_over, min_displacement=0.03) == 0.0  # bearing is 0 rad here anyway
    just_over_lateral = np.array([0.0, 0.05, 0.0])
    assert abs(compute_heading_err(just_over_lateral, min_displacement=0.03) - np.pi / 2) < 1e-9


if __name__ == "__main__":
    test_heading_err_ignores_near_zero_displacement_noise()
    test_heading_err_trusts_real_displacement()
    test_heading_err_threshold_boundary()

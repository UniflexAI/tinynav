import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tool.stair_odom_noise import OdomNoise, OdomNoiseConfig


def test_timestamp_lookup_is_repeatable_across_stream_rates_and_order():
    cfg = OdomNoiseConfig(mode='combined')
    a, b = OdomNoise(cfg, 5), OdomNoise(cfg, 5)
    expected = a.error(1.113)
    for t in np.arange(0, 2, .02):
        b.error(t)
    np.testing.assert_array_equal(expected, b.error(1.113))
    np.testing.assert_array_equal(expected, a.error(1.113))
    assert not np.array_equal(expected, OdomNoise(OdomNoiseConfig(mode='combined', seed=8), 5).error(1.113))


@pytest.mark.parametrize('cfg', [OdomNoiseConfig(), OdomNoiseConfig(mode='combined', scale=0)])
def test_disabled_noise_preserves_pose_exactly_and_never_mutates_input(cfg):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [21, 13, 8], degrees=True).as_matrix()
    T[:3, 3] = [1, 2, 3]
    original = T.copy()
    result = OdomNoise(cfg, 5).apply(T, 1.5)
    np.testing.assert_array_equal(T, original)
    np.testing.assert_array_equal(result, original)
    assert result is not T


def test_optical_pitch_and_world_yaw_are_applied_to_the_expected_axes():
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    cfg = OdomNoiseConfig(mode='pitch', pitch_std_deg=0, pitch_amplitude_deg=2, pitch_hz=.25)
    out = OdomNoise(cfg, 5).apply(T, 1.)
    assert np.degrees(np.arcsin(out[2, 2])) == pytest.approx(2.)
    np.testing.assert_allclose(out[:3, :3].T @ out[:3, :3], np.eye(3), atol=1e-14)
    np.testing.assert_array_equal(out[:3, 3], T[:3, 3])
    yaw = OdomNoise(OdomNoiseConfig(mode='yaw'), 5)
    error = yaw.error(1)[3]
    out = yaw.apply(T, 1)
    assert np.degrees(np.arctan2(out[1, 2], out[0, 2])) == pytest.approx(error)


def test_correlated_samples_have_configured_long_run_scale_and_zero_start():
    cfg = OdomNoiseConfig(mode='combined', pitch_amplitude_deg=0)
    noise = OdomNoise(cfg, 1000)
    samples = np.array([noise.error(t) for t in np.arange(2, 1000, .02)])
    np.testing.assert_allclose(samples.std(axis=0), [.01, .01, .01, .5, .5], rtol=.08)
    np.testing.assert_array_equal(noise.error(0), np.zeros(5))
    assert np.corrcoef(samples[:-1, 0], samples[1:, 0])[0, 1] > .85


@pytest.mark.parametrize('kwargs', [{'scale': -1}, {'pitch_hz': float('nan')}, {'correlation_s': 0}])
def test_invalid_noise_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OdomNoise(OdomNoiseConfig(**kwargs), 5)

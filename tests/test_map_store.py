import os
import shelve
import subprocess
import sys
import tempfile

import numpy as np

from tinynav.core.map_store import decode_depth, encode_depth, half_descriptors

TOOL = os.path.join(os.path.dirname(__file__), '..', 'tool', 'slim_map.py')


def _looper_depth(rng, shape=(48, 64)):
    """What looper_bridge_node publishes: the camera's uint16 mm over 1000, some of it
    saturated (65.535 m, just over 65535 when float32 multiplies it back)."""
    mm = rng.integers(0, 20000, size=shape).astype(np.uint16)
    mm[0, :3] = np.iinfo(np.uint16).max
    return mm.astype(np.float32) / 1000.0


def _superpoint(rng, n=32):
    descps = rng.standard_normal((1, n, 256)).astype(np.float16).astype(np.float32)
    return {'kpts': rng.random((1, n, 2), dtype=np.float32) * 100, 'descps': descps,
            'scores': rng.random((1, n), dtype=np.float32), 'mask': np.ones((1, n, 1), dtype=bool)}


def test_depth_on_a_mm_grid_comes_back_bit_identical():
    depth = _looper_depth(np.random.default_rng(0))
    stored = encode_depth(depth)
    assert stored.dtype == np.uint16 and stored.nbytes * 2 == depth.nbytes
    decoded = decode_depth(stored)
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, depth)


def test_other_depth_within_half_a_millimetre():
    depth = np.random.default_rng(1).uniform(0.2, 30.0, size=(48, 64)).astype(np.float32)
    assert np.abs(decode_depth(encode_depth(depth)) - depth).max() <= 0.0005 + 1e-6


def test_unusable_depth_is_stored_as_none():
    depth = np.array([[np.nan, np.inf, -np.inf], [0.0, -1.0, 70.0], [65.535, 65.536, 1.0]], dtype=np.float32)
    decoded = decode_depth(encode_depth(depth))
    expected = np.array([[0, 0, 0], [0, 0, 0], [65.535, 0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(decoded, expected, atol=1e-6)


def test_float32_depth_of_an_older_map_comes_back_as_it_is():
    depth = np.random.default_rng(2).uniform(0.2, 30.0, size=(4, 4)).astype(np.float32)
    assert decode_depth(depth) is depth


def test_descriptors_exact_in_fp16_are_halved():
    features = _superpoint(np.random.default_rng(3))
    half = half_descriptors(features)
    assert half['descps'].dtype == np.float16
    assert np.array_equal(half['descps'].astype(np.float32), features['descps'])
    assert all(half[k] is features[k] for k in ('kpts', 'scores', 'mask'))


def test_descriptors_not_exact_in_fp16_stay_float32():
    features = _superpoint(np.random.default_rng(4))
    features['descps'] = features['descps'] + np.float32(1e-6)
    assert half_descriptors(features) is features


def test_halved_descriptors_are_left_alone():
    half = half_descriptors(_superpoint(np.random.default_rng(5)))
    assert half_descriptors(half) is half


def _write_old_map(map_dir, rng, n=5):
    np.save(os.path.join(map_dir, 'poses.npy'), {}, allow_pickle=True)
    depths, features = {}, {}
    with shelve.open(os.path.join(map_dir, 'depths')) as d, shelve.open(os.path.join(map_dir, 'features')) as f, \
            shelve.open(os.path.join(map_dir, 'embeddings')) as e:
        for i in range(n):
            key = str(1000 + i)
            depths[key] = d[key] = _looper_depth(rng)
            features[key] = f[key] = {**_superpoint(rng), 'flexglue_v0_1': {
                'features': rng.standard_normal((1, 8, 4, 4)).astype(np.float16), 'image_size': (64, 48)}}
            e[key] = rng.standard_normal(8).astype(np.float32)
    return depths, features


def _slim(map_dir, *args):
    return subprocess.run([sys.executable, TOOL, map_dir, *args], check=True, capture_output=True, text=True).stdout


def test_slim_map_rewrites_a_map_and_keeps_what_it_holds():
    with tempfile.TemporaryDirectory() as map_dir:
        depths, features = _write_old_map(map_dir, np.random.default_rng(6))
        before = {f: os.path.getmtime(os.path.join(map_dir, f)) for f in os.listdir(map_dir)}

        _slim(map_dir, '--dry-run')
        assert {f: os.path.getmtime(os.path.join(map_dir, f)) for f in os.listdir(map_dir)} == before

        _slim(map_dir)
        assert sorted(os.listdir(map_dir)) == sorted(before), 'files were added or left behind'
        with shelve.open(os.path.join(map_dir, 'depths'), flag='r') as d:
            assert sorted(d.keys()) == sorted(depths)
            for key, depth in depths.items():
                assert d[key].dtype == np.uint16
                assert np.array_equal(decode_depth(d[key]), depth)
        with shelve.open(os.path.join(map_dir, 'features'), flag='r') as f:
            for key, original in features.items():
                stored = f[key]
                assert stored['descps'].dtype == np.float16
                assert np.array_equal(stored['descps'].astype(np.float32), original['descps'])
                for k in ('kpts', 'scores', 'mask'):
                    assert np.array_equal(stored[k], original[k])
                assert np.array_equal(stored['flexglue_v0_1']['features'], original['flexglue_v0_1']['features'])
        written = {f: os.path.getmtime(os.path.join(map_dir, f)) for f in os.listdir(map_dir)}

        assert '0/5 entries to re-encode' in _slim(map_dir)
        assert {f: os.path.getmtime(os.path.join(map_dir, f)) for f in os.listdir(map_dir)} == written


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'{name}: ok')

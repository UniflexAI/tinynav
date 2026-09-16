"""Preview frame rate is what caps the preview's bitrate.

JPEG frames are encoded independently, so the stream costs frame_size * fps
with no inter-frame saving -- measured on a rig, 320px/q50 runs 715 kbps at
15 fps regardless of what the camera sees. TINYNAV_PREVIEW_MAX_FPS is the lever
for a relayed uplink that cannot carry that.

Usage:
    cd /tinynav
    python tests/test_preview_frame_rate.py
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
import traceback
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.backend import node_manager
from app.backend.node_manager import BackendNode

TOPIC = '/camera/camera/infra1/image_rect_raw'


class _Msg:
    """A mono8 Image the preview path can decode."""

    encoding = 'mono8'
    height = 8
    width = 8
    data = np.zeros(8 * 8, dtype=np.uint8).tobytes()


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _bare_node(emitted):
    node = BackendNode.__new__(BackendNode)
    node._lock = threading.Lock()
    node.preview_callbacks = {TOPIC: [(lambda frame: emitted.append(frame), 320, 50)]}
    node._last_frame = {}
    node._last_frame_time = {}
    return node


def _feed(fps_cap, source_hz, seconds):
    """Drive the preview path at source_hz for `seconds` of simulated time."""
    emitted = []
    node = _bare_node(emitted)
    clock = _Clock()
    real_time, real_interval = node_manager.time, node_manager._PREVIEW_MIN_INTERVAL
    node_manager.time = SimpleNamespace(time=clock)
    node_manager._PREVIEW_MIN_INTERVAL = 1.0 / fps_cap
    try:
        for _ in range(int(source_hz * seconds)):
            node._on_image(_Msg(), TOPIC)
            clock.now += 1.0 / source_hz
    finally:
        node_manager.time = real_time
        node_manager._PREVIEW_MIN_INTERVAL = real_interval
    return emitted


def test_cap_holds_the_emitted_rate_down():
    seconds = 4.0
    emitted = _feed(fps_cap=5, source_hz=30, seconds=seconds)
    rate = len(emitted) / seconds
    assert 4.0 <= rate <= 6.0, (
        f'a 5 fps cap let {rate:.1f} fps through from a 30 Hz source; the '
        f'preview bitrate scales with this rate, so the cap is the bitrate')


def test_frames_spaced_beyond_the_cap_all_pass():
    # The companion to the test above: a throttle that dropped everything, or
    # one wired to the wrong clock, would satisfy the cap and starve the viewer.
    seconds = 4.0
    emitted = _feed(fps_cap=5, source_hz=5, seconds=seconds)
    assert len(emitted) >= int(5 * seconds) - 1, (
        f'a 5 Hz source under a 5 fps cap yielded only {len(emitted)} frames in '
        f'{seconds}s; the throttle is dropping frames it should pass')


def test_env_sets_the_interval():
    for value, expected in (('5', 0.2), ('20', 0.05)):
        os.environ['TINYNAV_PREVIEW_MAX_FPS'] = value
        try:
            reloaded = importlib.reload(node_manager)
            assert abs(reloaded._PREVIEW_MIN_INTERVAL - expected) < 1e-9, (
                f'TINYNAV_PREVIEW_MAX_FPS={value} gave interval '
                f'{reloaded._PREVIEW_MIN_INTERVAL}, expected {expected}')
        finally:
            del os.environ['TINYNAV_PREVIEW_MAX_FPS']
    reloaded = importlib.reload(node_manager)
    assert abs(reloaded._PREVIEW_MIN_INTERVAL - 0.05) < 1e-9, (
        'the default must stay 20 fps for rigs on a LAN')


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_'):
            continue
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception:
            failures += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    print('FAILED' if failures else 'OK')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

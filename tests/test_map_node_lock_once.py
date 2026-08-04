"""Optional lock-once relocalization (TINYNAV_RELOC_LOCK_WINDOW).

`_obs_spread` is exercised as an unbound method against a stub, so no node (and no
TRT engine) is constructed; importing map_node still needs the ROS deps, which is
why this lives apart from the import-light test_map_node.py.

Default is 0 — upstream's continuous fusion, where none of this runs.
"""
import sys

sys.path.append(".")

import numpy as np

from tinynav.core.map_node import MapNode


class _Stub:
    """Just the attributes _obs_spread touches."""

    def __init__(self, translations, lock_window=3):
        self._reloc_obs_window = []
        for i, t in enumerate(translations):
            T = np.eye(4)
            T[:3, 3] = t
            self._reloc_obs_window.append((i, T))
        self.reloc_lock_window = lock_window


def test_window_not_full_yet():
    stub = _Stub([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)], lock_window=3)
    assert MapNode._obs_spread(stub) is None


def test_spread_is_the_max_pairwise_translation_distance():
    stub = _Stub([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.3, 0.0, 0.0)], lock_window=3)
    assert MapNode._obs_spread(stub) == 0.3


def test_a_lone_outlier_in_the_window_blocks_the_lock():
    # The point of pairing the agreement check with the freeze: locking is permanent
    # for the route, so a burst containing a ~5.7m mis-match must not lock.
    stub = _Stub([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (5.7, 0.0, 0.0)], lock_window=3)
    assert MapNode._obs_spread(stub) > 0.3


def test_a_window_of_one_locks_on_whatever_arrives():
    # Documented, not recommended: nothing to disagree with.
    stub = _Stub([(5.7, 0.0, 0.0)], lock_window=1)
    assert MapNode._obs_spread(stub) == 0.0

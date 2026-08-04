"""MapNode's relocalization burst gate.

`_burst_spread` is exercised as an unbound method against a stub, so no node (and
no TRT engine) is constructed; importing map_node still needs the ROS deps, which
is why this lives apart from the import-light test_map_node.py.
"""
import sys

sys.path.append(".")

import numpy as np

from tinynav.core.map_node import MapNode


class _Stub:
    """Just the attributes _burst_spread touches."""

    def __init__(self, translations, burst_window=3, bootstrap_window=1):
        self._reloc_obs_window = []
        for t in translations:
            T = np.eye(4)
            T[:3, 3] = t
            self._reloc_obs_window.append(T)
        self.reloc_burst_window = burst_window
        self.reloc_bootstrap_window = bootstrap_window


def test_single_observation_agrees_with_itself():
    # A one-observation window has no pairs. Before this returned 0.0, the pairwise
    # np.max() below it raised ValueError on every keyframe, which meant a
    # bootstrap_window of 1 could never produce a fix at all.
    stub = _Stub([(1.0, 2.0, 3.0)])
    assert MapNode._burst_spread(stub, 1) == 0.0


def test_window_not_full_yet():
    stub = _Stub([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0)])
    assert MapNode._burst_spread(stub, 3) is None
    assert MapNode._burst_spread(stub) is None       # defaults to reloc_burst_window


def test_spread_is_the_max_pairwise_translation_distance():
    stub = _Stub([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.3, 0.0, 0.0)])
    assert MapNode._burst_spread(stub) == 0.3


def test_only_the_last_n_observations_count():
    # The sliding window holds reloc_burst_window entries for the kidnap check; the
    # bootstrap check must look at the most recent ones, not the whole window.
    stub = _Stub([(9.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.1, 0.0, 0.0)])
    assert MapNode._burst_spread(stub, 2) == 0.1
    assert MapNode._burst_spread(stub, 3) == 9.0


def test_bootstrap_takes_the_first_observation():
    # Bootstrap is a window of 1: one observation, nothing to disagree with. What a
    # burst of any size could not have caught anyway is a persistently mis-matched
    # place — observations of it agree with each other, tightly.
    stub = _Stub([(5.8, 0.0, 0.0), (5.82, 0.0, 0.0)])
    assert MapNode._burst_spread(stub, 1) <= 0.3
    assert MapNode._burst_spread(stub, 2) <= 0.3     # a 2-burst would accept it too


def test_bootstrap_and_kidnap_windows_are_independent():
    # bootstrap (1) trusts the newest observation on its own, while the kidnap check
    # (3) still sees a wide spread and so cannot license throwing away a working
    # estimate on the strength of one outlier.
    stub = _Stub([(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (5.0, 0.0, 0.0)])
    tol = 0.3
    assert MapNode._burst_spread(stub, stub.reloc_bootstrap_window) <= tol
    assert MapNode._burst_spread(stub, stub.reloc_burst_window) > tol

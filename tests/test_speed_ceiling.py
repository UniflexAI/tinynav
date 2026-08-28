"""The forward-speed ceiling, and the reduction applied on stairs.

vx_hard_max is an OPERATIONAL limit, deliberately below what the chassis can do (go2w
runs to 1.5 m/s), so what matters is not its value but that every path setting a target
speed is bounded by it -- including the capture-speed prior, which is the one input
allowed to raise the target above vx_max.

The methods are extracted from the source and run against a stand-in `self`, so these
assertions bind to planning_node's real arithmetic rather than to a copy of it:
planning_node needs numba and ROS at import (same approach as test_trajectory_caps.py).
"""
from __future__ import annotations

import ast
import os
import types
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLANNING = os.path.join(_HERE, '..', 'tinynav', 'core', 'planning_node.py')
_SRC = ast.parse(open(_PLANNING).read())

_WANTED = {'_open_target_speed', '_speed_from_clearance', '_on_stairs_now',
           '_signal_fresh'}


def _methods():
    """The real method objects, lifted off PlanningNode without importing it."""
    cls = next(n for n in ast.walk(_SRC)
               if isinstance(n, ast.ClassDef) and n.name == 'PlanningNode')
    fns = [n for n in cls.body
           if isinstance(n, ast.FunctionDef) and n.name in _WANTED]
    missing = _WANTED - {f.name for f in fns}
    assert not missing, f'not found on PlanningNode: {sorted(missing)}'
    mod = ast.Module(body=fns, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict = {'np': np}
    exec(compile(mod, '<planning>', 'exec'), ns)
    return {name: ns[name] for name in _WANTED}


def _declared_default(name):
    for node in ast.walk(_SRC):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'declare_parameter' and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == name):
            return ast.literal_eval(node.args[1])
    raise AssertionError(f'declare_parameter({name!r}) not found')


_M = _methods()
CEILING = _declared_default('vx_hard_max')
VX_MIN = _declared_default('vx_min')
VX_MAX = _declared_default('vx_max')
STAIRS_SCALE = _declared_default('stairs_speed_scale')


class _Node:
    """Stand-in `self`: just the attributes the extracted methods read."""

    def __init__(self, speed_cap=None, on_stairs=False, fresh=True, gain=1.0,
                 stairs_scale=STAIRS_SCALE):
        self._vx_min, self._vx_max, self._vx_hard_max = VX_MIN, VX_MAX, CEILING
        self._capture_speed_gain = gain
        self._speed_cap = speed_cap
        self._speed_cap_stamp_ns = 0 if (speed_cap is not None and fresh) else None
        self._speed_cap_ttl_ns = int(2e9)
        self._stairs_speed_scale = stairs_scale
        self._on_stairs = on_stairs
        self._on_stairs_stamp_ns = 0 if on_stairs else None
        self._on_stairs_ttl_ns = int(2e9)
        self._t_react_s, self._clear_c0_m, self._clear_open_m = 0.2, 0.35, 1.0
        for name, fn in _M.items():
            setattr(self, name, types.MethodType(fn, self))

    def get_clock(self):
        # Now == the stamps above, so a set stamp reads fresh and None reads stale.
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=0))

    def stale_stairs(self):
        self._on_stairs_stamp_ns = -int(1e12)   # long past the TTL
        return self


class CeilingTest(unittest.TestCase):
    def test_the_capture_prior_cannot_raise_the_target_past_the_ceiling(self):
        # The prior is explicitly allowed to exceed vx_max; the ceiling is what stops it.
        for cap in (0.5, 1.0, 1.4, 3.0, 1e6):
            for gain in (1.0, 2.0):
                v = _Node(speed_cap=cap, gain=gain)._open_target_speed()
                self.assertLessEqual(v, CEILING + 1e-9, f'cap={cap} gain={gain}')

    def test_a_stale_or_unknown_prior_falls_back_under_the_ceiling(self):
        self.assertLessEqual(
            _Node(speed_cap=float('nan'))._open_target_speed(), CEILING)
        self.assertLessEqual(
            _Node(speed_cap=2.0, fresh=False)._open_target_speed(), CEILING)

    def test_the_clearance_schedule_never_exceeds_the_target_it_was_given(self):
        node = _Node(speed_cap=1e6)
        v_open = node._open_target_speed()
        for clearance in (0.0, 0.35, 1.0, 2.0, 50.0):
            v = node._speed_from_clearance(clearance, 0.0, v_open)
            self.assertLessEqual(v, CEILING + 1e-9, f'clearance={clearance}')

    def test_the_fallback_target_is_itself_under_the_ceiling(self):
        # vx_max is the no-prior target, so a ceiling below it would never be reached.
        self.assertLessEqual(VX_MAX, CEILING + 1e-9)

    def test_the_ceiling_leaves_room_above_the_creep_speed(self):
        # A ceiling at or under vx_min would collapse the whole schedule onto one speed.
        self.assertLess(VX_MIN, CEILING)

    # The other constraint on this number -- that core_runtime's vio_guard still has
    # margin over it before calling real driving a pose jump -- is asserted where
    # vio_guard lives: core_runtime/tests/test_vio_guard_odom.py. Pinning a value here
    # would only restate the parameter back to itself.


class StairsSpeedTest(unittest.TestCase):
    def test_stairs_scale_the_target_by_the_parameter(self):
        # Injects a scale rather than reading the default: the default is 1.0 today, and
        # a test written against it would pass with the multiplication deleted.
        for cap in (None, 0.8, CEILING):
            flat = _Node(speed_cap=cap)._open_target_speed()
            stairs = _Node(speed_cap=cap, on_stairs=True,
                           stairs_scale=0.5)._open_target_speed()
            self.assertAlmostEqual(stairs, max(flat * 0.5, VX_MIN), places=6,
                                   msg=f'speed_cap={cap}')

    def test_the_reduction_reaches_the_schedule_too(self):
        # Applied inside _open_target_speed so the lattice bound and the published cap
        # cannot disagree: open ground on stairs must saturate at the reduced target.
        node = _Node(speed_cap=CEILING, on_stairs=True, stairs_scale=0.5)
        self.assertAlmostEqual(node._speed_from_clearance(50.0, 0.0,
                                                          node._open_target_speed()),
                               CEILING * 0.5, places=6)

    def test_stairs_never_scale_below_the_creep_speed(self):
        # A slow capture speed on a staircase must not scale down into a freeze.
        v = _Node(speed_cap=VX_MIN)._open_target_speed()
        self.assertGreaterEqual(
            _Node(speed_cap=VX_MIN, on_stairs=True,
                  stairs_scale=0.1)._open_target_speed(), VX_MIN)
        self.assertGreaterEqual(v, VX_MIN)

    def test_a_stale_on_stairs_stream_does_not_slow_the_robot(self):
        # map_node going quiet is not evidence of stairs.
        node = _Node(speed_cap=CEILING, on_stairs=True).stale_stairs()
        self.assertAlmostEqual(node._open_target_speed(),
                               _Node(speed_cap=CEILING)._open_target_speed(), places=6)

    def test_off_stairs_is_unchanged(self):
        self.assertAlmostEqual(_Node(speed_cap=CEILING)._open_target_speed(),
                               CEILING, places=6)

    def test_the_scale_stays_within_its_meaning(self):
        # A scale is a reduction factor: never zero (a freeze) and never an amplifier.
        # 1.0 -- the current default -- is the neutral end of that range, not a bug.
        self.assertGreater(STAIRS_SCALE, 0.0)
        self.assertLessEqual(STAIRS_SCALE, 1.0)


if __name__ == '__main__':
    unittest.main()

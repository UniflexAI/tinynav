"""The lateral-acceleration cap on the lattice.

`duration` alone ties the turn rate to vx: the omega range a 3 s lattice offers at the
speed open ground allows (1.34 m/s measured on device) is the same one it offers at a
crawl. `max_lat_acc` caps vx*omega instead.

The trap is the turn-in-place rows. Those are what the cost function's heading term
ranks to get the robot facing a goal behind it, and vx*omega is 0 there, so the cap is
tested for what it bounds AND for leaving the stationary rows alone.

`planning_node` needs numba and ROS at import, so the lattice function is extracted and
compiled on its own (its @njit stripped) against numpy -- same idea as
core_runtime/tests/test_vio_guard_odom.py.
"""
from __future__ import annotations

import ast
import math
import os
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, '..', 'tinynav', 'core')


def _load():
    ns: dict = {'np': np}
    for path, names in (
        (os.path.join(_CORE, 'math_utils.py'),
         {'rotvec_to_matrix', 'quat_to_matrix', 'matrix_to_quat'}),
        (os.path.join(_CORE, 'planning_node.py'),
         {'generate_trajectory_library_3d'}),
    ):
        tree = ast.parse(open(path).read())
        fns = [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name in names]
        for f in fns:
            f.decorator_list = []          # drop @njit; the body is plain numpy
        mod = ast.Module(body=fns, type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, '<lattice>', 'exec'), ns)
    return ns['generate_trajectory_library_3d']


_gen = _load()

# Camera +Z forward along world +X: the lattice's z-flattening hack is about world
# height, so a heading that is level is the only one that leaves the arcs their shape.
_LEVEL = np.array([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])
_V_OPEN = 1.34        # v_allow measured in open ground on device
_OMEGA_MAX = 0.75     # ROBOT_CONFIG.max_angular_vel
_OFF = 1e9


def _lattice(max_lat_acc=_OFF):
    return _gen(init_p=np.zeros(3), init_q=_LEVEL,
                max_linear_vel=_V_OPEN, max_angular_vel=_OMEGA_MAX,
                max_lat_acc=max_lat_acc)


def _stationary(params):
    return np.abs(params[:, 0]) < 1e-9


class LateralAccelCapTest(unittest.TestCase):
    def test_uncapped_full_omega_is_offered_at_full_speed(self):
        _, params = _lattice()
        top = params[:, 0].max()
        self.assertAlmostEqual(
            np.abs(params[params[:, 0] > top - 1e-9, 1]).max(), _OMEGA_MAX, places=6)

    def test_vx_times_omega_stays_under_the_cap(self):
        for cap in (0.3, 0.5, 1.0):
            _, params = _lattice(max_lat_acc=cap)
            self.assertLessEqual(np.abs(params[:, 0] * params[:, 1]).max(),
                                 cap + 1e-9, f'cap={cap}')

    def test_it_bites_hardest_at_the_top_speed(self):
        _, params = _lattice(max_lat_acc=0.5)
        speeds = np.unique(params[:, 0])
        widest = [np.abs(params[params[:, 0] == v, 1]).max() for v in speeds]
        # Monotonically non-increasing in vx, and strictly smaller at the top.
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(widest, widest[1:])))
        self.assertLess(widest[-1], widest[0])

    def test_slow_rows_are_untouched(self):
        # Below max_lat_acc/max_angular_vel the cap must not bind at all.
        cap = 0.5
        _, free = _lattice()
        _, capped = _lattice(max_lat_acc=cap)
        for v in np.unique(capped[:, 0]):
            if v > cap / _OMEGA_MAX - 1e-9:
                continue
            self.assertAlmostEqual(np.abs(capped[capped[:, 0] == v, 1]).max(),
                                   np.abs(free[free[:, 0] == v, 1]).max(), places=6,
                                   msg=f'vx={v:.3f} should be below the knee')

    def test_turn_in_place_keeps_full_angular_rate(self):
        # vx*omega is 0 at a standstill, so no lateral-accel cap may narrow it.
        for cap in (0.1, 0.5):
            _, params = _lattice(max_lat_acc=cap)
            self.assertAlmostEqual(np.abs(params[_stationary(params), 1]).max(),
                                   _OMEGA_MAX, places=6, msg=f'cap={cap}')


class LatticeShapeTest(unittest.TestCase):
    def test_the_row_count_is_unchanged_by_the_cap(self):
        self.assertEqual(_lattice(max_lat_acc=0.3)[1].shape, _lattice()[1].shape)

    def test_every_trajectory_stays_level(self):
        trajs, _ = _lattice(max_lat_acc=0.5)
        for t in trajs:
            self.assertTrue(np.allclose(t[:, 2], t[0, 2]))


if __name__ == '__main__':
    unittest.main()

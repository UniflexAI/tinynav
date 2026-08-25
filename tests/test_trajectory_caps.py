"""The two caps that shape the lattice independently of the speed it is allowed.

`duration` alone ties both the planning horizon and the turn rate to vx: at the speed
open ground allows (1.34 m/s measured on device) a 3 s lattice reaches a 4 m arc, and
the omega range it offers at that speed is the same one it offers at a crawl.

Both caps have the same trap — the turn-in-place rows. Those are what the cost
function's heading term ranks to get the robot facing a goal behind it, and a cap that
shortens them re-opens the freeze that term was added to close. So each cap is tested
for what it bounds AND for leaving the stationary rows alone.

`planning_node` needs numba and ROS at import, so the lattice function is extracted and
compiled on its own (its @njit stripped) against numpy — same idea as
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
_DURATION = 3.0
_OFF = 1e9


def _lattice(max_path_len_m=_OFF, max_lat_acc=_OFF):
    return _gen(init_p=np.zeros(3), init_q=_LEVEL,
                max_linear_vel=_V_OPEN, max_angular_vel=_OMEGA_MAX,
                max_path_len_m=max_path_len_m, max_lat_acc=max_lat_acc)


def _arc_lengths(trajs):
    return np.array([np.linalg.norm(np.diff(t[:, :3], axis=0), axis=1).sum()
                     for t in trajs])


def _stationary(params):
    return np.abs(params[:, 0]) < 1e-9


def _swing(traj):
    """Angle actually turned from the first pose to the last, in radians.

    Read off the trajectory, NOT off params[:, 1]: the sampled omega is what the row
    was asked for, and stays put whatever the caps do, so asserting on it is asserting
    on nothing. The arc-length cap freezes a row's integration -- rotation included --
    so only the geometry can say whether a row still swings.
    """
    dot = abs(float(np.dot(traj[0, 3:], traj[-1, 3:])))
    return 2.0 * math.acos(min(1.0, dot))


class ArcLengthCapTest(unittest.TestCase):
    def test_uncapped_the_lattice_reaches_vx_times_duration(self):
        # The thing being fixed: nothing bounds how far ahead a plan commits.
        trajs, _ = _lattice()
        self.assertAlmostEqual(_arc_lengths(trajs).max(), _V_OPEN * _DURATION,
                               delta=0.15)

    def test_no_trajectory_exceeds_the_cap(self):
        for cap in (1.0, 2.5, 3.5):
            trajs, _ = _lattice(max_path_len_m=cap)
            self.assertLessEqual(_arc_lengths(trajs).max(), cap + 1e-6, f'cap={cap}')

    def test_a_tighter_cap_draws_shorter(self):
        long_ = _arc_lengths(_lattice(max_path_len_m=3.0)[0]).max()
        short = _arc_lengths(_lattice(max_path_len_m=1.5)[0]).max()
        self.assertLess(short, long_)

    def test_the_cap_does_not_slow_the_robot_down(self):
        # Speed is the feedforward; the cap is about how far the shape is drawn, so
        # the vx vocabulary has to survive it intact.
        _, free = _lattice()
        _, capped = _lattice(max_path_len_m=1.0)
        np.testing.assert_allclose(np.unique(capped[:, 0]), np.unique(free[:, 0]))

    def test_turn_in_place_keeps_its_full_swing(self):
        # vx=0 accumulates no length, so the cap must not touch these rows -- they
        # are what the cost function's heading term ranks, and a shortened swing
        # re-opens the standstill freeze that term exists to close.
        trajs, params = _lattice()
        free = max(_swing(t) for t in trajs[_stationary(params)])
        self.assertAlmostEqual(free, _OMEGA_MAX * _DURATION, delta=0.1)
        for cap in (0.3, 0.5, 1.0, 2.5):
            trajs, params = _lattice(max_path_len_m=cap)
            capped = max(_swing(t) for t in trajs[_stationary(params)])
            self.assertAlmostEqual(capped, free, places=6,
                                   msg=f'cap={cap} shortened the turn-in-place rows')

    def test_a_moving_row_does_stop_turning_once_it_hits_the_cap(self):
        # The other side of the same property: freezing is real, and it is the
        # moving rows it applies to.
        trajs, params = _lattice(max_path_len_m=0.5)
        moving = ~_stationary(params) & (np.abs(params[:, 1]) > 0.1)
        free_trajs, free_params = _lattice()
        free_moving = ~_stationary(free_params) & (np.abs(free_params[:, 1]) > 0.1)
        self.assertLess(max(_swing(t) for t in trajs[moving]),
                        max(_swing(t) for t in free_trajs[free_moving]))


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
    def test_the_row_count_is_unchanged_by_either_cap(self):
        base = _lattice()[1].shape
        for kwargs in ({'max_path_len_m': 1.0}, {'max_lat_acc': 0.3},
                       {'max_path_len_m': 1.0, 'max_lat_acc': 0.3}):
            self.assertEqual(_lattice(**kwargs)[1].shape, base, str(kwargs))

    def test_every_trajectory_stays_level(self):
        trajs, _ = _lattice(max_path_len_m=2.5, max_lat_acc=0.5)
        for t in trajs:
            self.assertTrue(np.allclose(t[:, 2], t[0, 2]))


if __name__ == '__main__':
    unittest.main()

"""Which constraints the solve is handed.

No ROS: the function is handed everything it reads, which is why it is a plain
function of two lists (tinynav/core/fusion_window.py).
"""
import unittest
from unittest import mock

import numpy as np

from tinynav.core import fusion_window
from tinynav.core.fusion_window import select_fusion_constraints


def _odom(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


def _constraint(implied_x: float) -> tuple:
    """One constraint, carrying the map->odom it implies. The solver reads that 4x4;
    the tests read it back to say which survived, since a tuple holding a numpy array
    is not comparable and `list.index` cannot."""
    return (0, 1, _odom(implied_x))


def _pick(xs, implied=None):
    """`xs` is where each constraint was observed; `implied` what each one says the
    transform is. Default: they all agree, so only staleness is under test."""
    implied = [0.0] * len(xs) if implied is None else implied
    cons = [_constraint(v) for v in implied]
    kept = select_fusion_constraints(cons, [_odom(x) for x in xs])
    return [round(float(c[2][0, 3]), 3) for c in kept]


class StalenessTest(unittest.TestCase):
    def test_a_constraint_from_before_the_last_few_metres_is_dropped(self):
        """An observation's implied map->odom is only still true while odom has not
        drifted since, and drift comes with travel. Measured on 118, keeping two
        minutes of them held a wrong estimate 0.75m from what the camera was saying."""
        far = fusion_window.FUSE_MAX_M + 5.0
        xs = [far] * 10 + [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # the stale ten say something else, so surviving would be visible
        implied = [9.0] * 10 + [0.0] * 6
        self.assertEqual(_pick(xs, implied), [0.0] * 6)

    def test_standing_still_expires_nothing(self):
        """A stationary robot travels nowhere, so its constraints stay valid however
        long it stands there -- otherwise standing still would starve the solve."""
        self.assertEqual(len(_pick([0.0] * 40)), 40)

    def test_the_count_is_still_capped(self):
        """Standing still for an hour must not grow the solve without bound."""
        self.assertEqual(len(_pick([0.0] * (fusion_window.FUSE_MAX + 50))),
                         fusion_window.FUSE_MAX)

    def test_staleness_is_cut_before_agreement_is_measured(self):
        """The map's own keyframe poses are a VIO trajectory, so it is locally
        consistent and globally warped: honest constraints from stretches far apart
        imply genuinely different transforms. Measuring agreement first would read that
        as disagreement and could hand the vote to the older stretch."""
        far = fusion_window.FUSE_MAX_M + 5.0
        xs = [far] * 9 + [0.0] * 5
        implied = [7.0] * 9 + [0.0] * 5          # the old majority says something else
        self.assertEqual(_pick(xs, implied), [0.0] * 5,
                         'an older, larger, self-consistent stretch won the vote')


class ConsensusTest(unittest.TestCase):
    def test_a_minority_that_disagrees_loses_the_vote(self):
        """The corridor aliasing, answered where it can be: a match from the far end
        of a self-similar corridor is a minority report, and nothing has to judge it."""
        implied = [0.0, 0.1, 0.05, 49.8, 49.7, 0.0, 0.1]
        self.assertEqual(_pick([0.0] * 7, implied), [0.0, 0.1, 0.05, 0.0, 0.1])

    def test_no_majority_means_the_pose_does_not_move(self):
        """An empty list is handed back and the solver returns the transform it was
        given -- checked against Ceres, which converges on zero residuals and leaves
        `optimized_parameters[0]` untouched. Odom carries the pose until a majority
        forms, which is what odom is for."""
        implied = [0.0, 9.0, 18.0, 27.0, 36.0, 45.0]
        self.assertEqual(_pick([0.0] * 6, implied), [])

    def test_a_short_history_cannot_form_a_majority(self):
        """Below the floor there are observations but no majority, and one observation
        is one PnP with no averaging left to damp it."""
        self.assertEqual(_pick([0.0] * (fusion_window.FUSE_MIN - 1)), [])

    def test_agreement_is_about_the_transform_not_where_it_was_observed(self):
        """All seven were taken in the same place; five agree about where that place
        is on the map. Standing still must not make an aliased match agree."""
        implied = [0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 30.0]
        self.assertEqual(_pick([0.0] * 7, implied), [0.0] * 5)

    def test_an_even_split_is_not_a_majority(self):
        """Half saying one thing and half another is exactly the case where nothing
        should be believed, and the robust line lands between them so neither side is
        on it. The pose stays where odom put it."""
        with mock.patch.object(fusion_window, 'FUSE_MIN', 3):
            implied = [0.0, 0.0, 0.0, 20.0, 20.0, 20.0]
            self.assertEqual(_pick([0.0] * 6, implied), [])


class TrendTest(unittest.TestCase):
    """The transform moves across a window -- the map's own keyframe poses are a VIO
    trajectory, so it is locally consistent and globally warped, and the true transform
    slides as the robot travels through that warp."""

    def test_a_drifting_trend_is_kept_end_to_end(self):
        """A cluster was tried first and measured dropping 78% of the constraints on
        122, the nearest one 0.01m from those it kept: it was cutting a continuum, not
        rejecting aliasing. The span here is 1.9m, far outside any agreement radius,
        and every point is still on the line."""
        xs = [0.1 * i for i in range(20)]
        implied = [1.0 * x for x in xs]            # 1m of transform per metre driven
        self.assertEqual(len(_pick(xs, implied)), 20)

    def test_an_aliased_block_is_still_dropped_from_a_drifting_trend(self):
        """The line must not tilt to fit them, which is why it is a median of pairwise
        slopes and not least squares."""
        xs = [0.1 * i for i in range(20)]
        implied = [1.0 * x for x in xs]
        for i in (7, 8, 12):
            implied[i] += 40.0
        kept = _pick(xs, implied)
        self.assertEqual(len(kept), 17)
        self.assertTrue(all(v < 10.0 for v in kept), 'an aliased match rode the line in')


if __name__ == '__main__':
    unittest.main()

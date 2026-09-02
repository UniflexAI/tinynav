"""Which constraints the solve is handed.

No ROS: the function is handed everything it reads, which is why it is a plain
function of two lists (tinynav/core/fusion_window.py).
"""
import unittest

import numpy as np

from tinynav.core import fusion_window
from tinynav.core.fusion_window import select_fusion_constraints


def _odom(x: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    return T


def _constraint(i: int) -> tuple:
    """One constraint, tagged by its index. The solver never looks past the first two
    fields, so the tag rides where the 4x4 goes and the test can say which survived --
    `list.index` cannot, because a tuple holding a numpy array is not comparable."""
    return (0, 1, i)


def _pick(xs):
    cons = [_constraint(i) for i in range(len(xs))]
    return [c[2] for c in select_fusion_constraints(cons, [_odom(x) for x in xs])]


class FusionWindowTest(unittest.TestCase):
    def test_a_constraint_from_before_the_last_few_metres_is_dropped(self):
        """The whole point: an observation's implied map->odom is only still true
        while odom has not drifted since, and drift comes with travel. Measured on
        118, keeping two minutes of them held a wrong estimate 0.75m from what the
        camera was saying."""
        far = fusion_window.FUSE_MAX_M + 5.0
        # ten stale ones metres back, then six taken right here
        xs = [far] * 10 + [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual(_pick(xs), [10, 11, 12, 13, 14, 15])

    def test_standing_still_expires_nothing(self):
        """A stationary robot travels nowhere, so its constraints stay valid however
        long it stands there -- otherwise standing still would starve the solve."""
        self.assertEqual(len(_pick([0.0] * 40)), 40)

    def test_it_never_solves_over_fewer_than_the_floor(self):
        """One constraint is that observation alone, with no averaging left to damp an
        aliased match -- so a long drive with nothing recent still solves over the
        newest few rather than the newest one."""
        kept = _pick([i * 10.0 for i in range(20)])   # every one is far from the next
        self.assertEqual(len(kept), fusion_window.FUSE_MIN)
        self.assertEqual(kept[-1], 19, 'the newest must be among them')

    def test_the_count_is_still_capped(self):
        """Standing still for an hour must not grow the solve without bound."""
        self.assertEqual(len(_pick([0.0] * (fusion_window.FUSE_MAX + 50))),
                         fusion_window.FUSE_MAX)

    def test_a_short_history_is_passed_straight_through(self):
        """Below the floor there is nothing to choose, and no arithmetic worth doing
        on one odom pose."""
        cons = [_constraint(i) for i in range(3)]
        self.assertIs(select_fusion_constraints(cons, [_odom(0.0)] * 3), cons)


if __name__ == '__main__':
    unittest.main()

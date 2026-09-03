"""How many relocalizations the solve is handed.

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


def _pick(implied):
    """`implied` is what each constraint says the transform is, oldest first. Returns
    the surviving ones by that value, since a tuple holding a numpy array is not
    comparable and `list.index` cannot."""
    cons = [(0, 1, _odom(v)) for v in implied]
    kept = select_fusion_constraints(cons, [_odom(0.0)] * len(implied))
    return [round(float(c[2][0, 3]), 3) for c in kept]


class RecencyTest(unittest.TestCase):
    def test_only_the_newest_few_survive(self):
        with mock.patch.object(fusion_window, 'FUSE_WINDOW', 3):
            self.assertEqual(_pick([1.0, 2.0, 3.0, 4.0, 5.0]), [3.0, 4.0, 5.0])

    def test_a_long_history_does_not_outvote_the_present(self):
        """The reason the window is small at all. Against observations that ramp --
        which is what a globally warped map gives -- an average lags reality by half
        the window. Ninety-nine old ones saying 0 must not drag the answer off the
        three saying 9; measured on 118, that cost 0.75m while standing still."""
        with mock.patch.object(fusion_window, 'FUSE_WINDOW', 3):
            kept = _pick([0.0] * 99 + [9.0, 9.0, 9.0])
            self.assertEqual(kept, [9.0, 9.0, 9.0])

    def test_the_newest_observation_is_the_transform(self):
        """The default is 1: averaging is off, because the error this path sees is a
        flip between two PnP hypotheses and not noise. Averaging a bimodal
        distribution lands between the modes, and averaging it over a large window
        lands there AND behind the robot."""
        self.assertEqual(fusion_window.FUSE_WINDOW, 1)
        self.assertEqual(_pick([1.0, 2.0, 3.0]), [3.0])

    def test_a_short_history_is_used_as_is(self):
        """Fewer observations than the window is not a reason to refuse to move --
        a travel-bounded window with a floor was tried in this file's place and
        starved the solve exactly where the fix rate collapses."""
        with mock.patch.object(fusion_window, 'FUSE_WINDOW', 3):
            self.assertEqual(_pick([4.0]), [4.0])

    def test_nothing_observed_yields_nothing(self):
        self.assertEqual(_pick([]), [])


if __name__ == '__main__':
    unittest.main()

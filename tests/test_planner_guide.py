import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from tinynav.core.planner_guide import local_detour_target, retained_esdf


class GuideTests(unittest.TestCase):
    def test_wall_detour(self):
        wall = np.zeros((60,60),dtype=bool)
        wall[25:28,:35] = True
        esdf = distance_transform_edt(~wall)*.05
        waypoint = local_detour_target(esdf,np.zeros(3),.05,np.array([.8,1.2]),np.array([2.,1.2,0.]),.15)
        self.assertIsNotNone(waypoint)
        self.assertGreater(waypoint[1],1.2)
        ix = (waypoint[:2]/.05).astype(int)
        self.assertGreaterEqual(esdf[tuple(ix)],.175)

    def test_retained_obstacle_survives_map_decay(self):
        esdf = np.ones((40,40),dtype=np.float32)
        merged = retained_esdf(esdf,np.zeros(3),.05,{(10,10):'blocked'},.1,2)
        self.assertEqual(merged[20,20],0)
        self.assertEqual(merged[18,20],0)
        self.assertGreater(merged[5,5],0)

    def test_start_inside_soft_margin_can_exit(self):
        esdf = np.ones((40,40),dtype=np.float32)
        esdf[10,10] = .15
        waypoint = local_detour_target(esdf,np.zeros(3),.05,np.array([.525,.525]),np.array([1.5,1.5,0.]),.15)
        self.assertIsNotNone(waypoint)

    def test_blocked_start(self):
        self.assertIsNone(local_detour_target(np.zeros((20,20)),np.zeros(3),.05,np.array([.5,.5]),np.array([.8,.8,0.]),.15))

    def test_sealed_wall(self):
        wall = np.zeros((40,40),dtype=bool)
        wall[20:23,:] = True
        esdf = distance_transform_edt(~wall)*.05
        waypoint = local_detour_target(esdf,np.zeros(3),.05,np.array([.5,1.]),np.array([1.5,1.,0.]),.15)
        if waypoint is not None:
            self.assertLess(waypoint[0],1.)


if __name__ == '__main__':
    unittest.main()

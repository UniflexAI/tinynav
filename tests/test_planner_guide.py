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


class GuideLifecycleTests(unittest.TestCase):
    def setUp(self):
        from tinynav.core.planner_guide import LocalGuide
        self.guide = LocalGuide()
        self.esdf = np.ones((80,80),dtype=np.float32)
        self.xy = np.array([.5,.5,0.])
        self.goal = np.array([2.,.5,0.])

    def update(self, now, **kwargs):
        return self.guide.update(self.xy,self.goal,now,self.esdf,np.zeros(3),.05,.15,{},.1,2,**kwargs)

    def test_stall_cache_and_reset(self):
        from unittest.mock import patch
        with patch('tinynav.core.planner_guide.retained_esdf',return_value=self.esdf) as build:
            self.assertIsNone(self.update(10))
            self.assertIsNotNone(self.update(12.1))
            self.assertIsNotNone(self.update(12.2))
            self.assertEqual(build.call_count,1)
            self.guide.reset()
            self.assertIsNone(self.update(12.3))
            self.assertIsNone(self.guide.waypoint)
            self.assertIsNotNone(self.update(14.4))
            self.assertEqual(build.call_count,2)

    def test_goal_change_restarts_stall_detection(self):
        self.update(10)
        self.assertIsNotNone(self.update(12.1))
        self.goal = np.array([.5,2.,0.])
        self.assertIsNone(self.update(12.2))
        self.assertIsNone(self.guide.waypoint)

    def test_recovery_ownership_invalidates_cache(self):
        from unittest.mock import patch
        with patch('tinynav.core.planner_guide.retained_esdf',return_value=self.esdf) as build:
            self.update(10)
            self.assertIsNotNone(self.update(12.1))
            self.assertIsNone(self.update(12.2,recovery_owned=True))
            self.assertEqual(build.call_count,1)
            self.assertIsNotNone(self.update(12.3))
            self.assertEqual(build.call_count,2)

    def test_navigation_callbacks_clear_guidance(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from tinynav.core.planning_node import PlanningNode
        for callback, message in [(PlanningNode.paused_callback,SimpleNamespace(data=True)),
                                  (PlanningNode.active_callback,SimpleNamespace(data=False)),
                                  (PlanningNode.poi_change_callback,None)]:
            with self.subTest(callback=callback.__name__):
                self.guide.reset()
                self.update(10)
                self.assertIsNotNone(self.update(12.1))
                node = SimpleNamespace(recovery=Mock(),guide=self.guide,target_pose=self.goal,
                    nav_active=True,nav_paused=False,recovery_target_anchor=self.goal.copy())
                node.reset_navigation_helpers = lambda target=None: PlanningNode.reset_navigation_helpers(node,target)
                callback(node,message)
                self.assertIsNone(self.guide.anchor)
                self.assertIsNone(self.guide.waypoint)
                self.assertEqual(self.guide.until,0)
                node.recovery.reset.assert_called_once()


if __name__ == '__main__':
    unittest.main()

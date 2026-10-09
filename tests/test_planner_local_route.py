import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from tinynav.core.planning_node import PlanningNode, local_detour_target, retained_esdf


class GuideTests(unittest.TestCase):
    def test_goal_outside_map_advances_along_corridor(self):
        wall = np.zeros((100,100),dtype=bool)
        wall[30:,62:70] = True
        esdf = distance_transform_edt(~wall)*.05
        waypoint = local_detour_target(esdf,np.array([-2.5,-2.5,0.]),.05,
                                      np.array([0.,0.,.45]),np.array([3.9,4.4,0.]),.15)
        self.assertIsNotNone(waypoint)
        self.assertGreater(waypoint[0],.2)
        self.assertLess(waypoint[1],.5)

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
        merged = retained_esdf(esdf,np.zeros(3),.05,{(10,10)},.1,2)
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


class LocalRouteTests(unittest.TestCase):
    def setUp(self):
        self.route = PlanningNode.__new__(PlanningNode)
        self.route.reset_local_route()
        self.map = np.ones((80,80),dtype=np.float32)
        self.xy = np.array([.5,.5,0.])
        self.goal = np.array([2.,.5,0.])

    def update(self, now):
        return self.route.update_local_route(self.xy,self.goal,now,self.map,np.zeros(3),.05,.15,0)

    def test_open_path_uses_original_planner(self):
        self.assertIsNone(self.update(10))

    def test_obstacle_routes_immediately_without_stall_timer(self):
        self.map[20:23,:15] = 0
        self.assertIsNotNone(self.update(10))

    def test_removed_obstacle_does_not_hold_detour(self):
        self.map[20:23,:15] = 0
        self.assertIsNotNone(self.update(10))
        self.map[:,:] = 1
        self.assertIsNone(self.update(10.6))

    def test_reset_discards_observed_obstacles(self):
        self.route.route_cells.add((1,1))
        self.route.reset_local_route()
        self.assertEqual(self.route.route_cells,set())
        self.assertIsNone(self.route.route_waypoint)

    def test_observation_retains_body_band_only(self):
        from types import SimpleNamespace
        robot = SimpleNamespace(obstacle=SimpleNamespace(robot_z_bottom=-.4,robot_z_top=.4))
        T = np.eye(4)
        K = np.eye(3)
        self.route.observe_route_obstacles(np.array([[1.]],dtype=np.float32),T,K,robot)
        self.assertFalse(self.route.route_cells)
        T[:3,:3] = np.array([[0,0,1],[1,0,0],[0,1,0]])
        self.route.observe_route_obstacles(np.array([[1.]],dtype=np.float32),T,K,robot)
        self.assertIn((10,0),self.route.route_cells)
        self.route.observe_route_obstacles(np.zeros((1,1),dtype=np.float32),T,K,robot)
        self.assertIn((10,0),self.route.route_cells)

    def test_navigation_changes_clear_route_cache_and_obstacles(self):
        from types import SimpleNamespace
        from tinynav.core.planning_node import PlanningNode
        for callback,message in [(PlanningNode.paused_callback,SimpleNamespace(data=True)),
                                 (PlanningNode.active_callback,SimpleNamespace(data=False)),
                                 (PlanningNode.poi_change_callback,None)]:
            with self.subTest(callback=callback.__name__):
                self.route.route_cells.add((1,1))
                self.route.route_waypoint = self.goal.copy()
                node = self.route
                node.nav_active, node.nav_paused, node.target_pose = True, False, self.goal
                callback(node,message)
                self.assertFalse(self.route.route_cells)
                self.assertIsNone(self.route.route_waypoint)

    def test_new_target_invalidates_route_before_observing(self):
        from types import SimpleNamespace
        from tinynav.core.planning_node import PlanningNode
        self.route.route_target = self.goal.copy()
        self.route.route_cells.add((1,1))
        node = self.route
        message = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=3.,y=2.,z=0.))))
        PlanningNode.target_pose_callback(node,message)
        self.assertFalse(self.route.route_cells)
        np.testing.assert_array_equal(node.target_pose,[3.,2.,0.])


if __name__ == '__main__':
    unittest.main()

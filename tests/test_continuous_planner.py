import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from tinynav.core.continuous_planner import guide_path, plan
from tinynav.core.robot_specs import GO2_CONFIG

class ContinuousPlannerTest(unittest.TestCase):
    def test_guide_routes_around_wall(self):
        occupied = np.zeros((100,100), bool)
        occupied[48:52, 15:65] = True
        esdf = distance_transform_edt(~occupied)*.05
        path = guide_path(np.array([1.,2.,0.]), np.array([4.,2.,0.]), esdf, np.zeros(3), .05, .45)
        self.assertIsNotNone(path)
        cells = np.floor(path/.05).astype(int)
        self.assertTrue(np.all(esdf[cells[:,0],cells[:,1]] >= .45))
        self.assertGreater(path[:,1].max(), 3.25)

    def test_open_rollout_and_velocity_bounds(self):
        q = Rotation.from_matrix(np.array([[0,0,1],[-1,0,0],[0,-1,0]])).as_quat()
        result, info = plan(np.array([0.,0.,0.]), q, np.array([1.,.4,0.]), np.full((100,100),10.), np.array([-2.5,-2.5,0.]), .05, GO2_CONFIG)
        self.assertIsNotNone(result)
        poses, cmd = result
        speed = np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)/.1
        yaw = np.unwrap(np.arctan2(Rotation.from_quat(poses[:,3:]).as_matrix()[:,1,2],Rotation.from_quat(poses[:,3:]).as_matrix()[:,0,2]))
        self.assertLessEqual(speed.max(),GO2_CONFIG.max_linear_vel+1e-6)
        self.assertLessEqual(np.abs(np.diff(yaw)/.1).max(),GO2_CONFIG.max_angular_vel+1e-6)
        self.assertLess(np.linalg.norm(poses[-1,:2]-[1.,.4]),np.linalg.norm([1.,.4]))

    def test_rectangular_body_fits_corridor(self):
        occupied = np.zeros((100, 100), bool)
        occupied[:, 43] = occupied[:, 57] = True
        esdf = distance_transform_edt(~occupied) * .05
        from dataclasses import replace
        robot = replace(GO2_CONFIG, safety_radius=.08)
        q = Rotation.from_matrix(np.array([[0,0,1],[-1,0,0],[0,-1,0]])).as_quat()
        output, _ = plan(np.array([1.,2.5,0.]), q, np.array([2.,2.5,0.]), esdf, np.zeros(3), .05, robot)
        self.assertIsNotNone(output)

    def test_sim_robot_parameters_override_preset(self):
        import subprocess, sys, os, json
        from dataclasses import asdict
        config = asdict(GO2_CONFIG)
        config['safety_radius'] = .08
        config['obstacle']['dilation_cells'] = 1
        env = dict(os.environ, TINYNAV_ROBOT_CONFIG=json.dumps(config))
        actual = subprocess.check_output([sys.executable, '-c',
            'from tinynav.core.robot_specs import ROBOT_CONFIG as r; print(r.safety_radius, r.obstacle.dilation_cells)'], env=env, text=True)
        self.assertEqual(actual.strip(), '0.08 1')

    def test_reverse_requires_observed_clear_body_sweep(self):
        from tinynav.core.continuous_planner import ContinuousPlanner
        from dataclasses import replace
        robot = replace(GO2_CONFIG, safety_radius=.08)
        q = Rotation.from_matrix(np.array([[0,0,1],[-1,0,0],[0,-1,0]])).as_quat()
        start, origin = np.zeros(3), np.array([-2.5,-2.5,0.])
        esdf = np.full((100,100),10.)
        unknown = np.zeros((100,100),bool)
        self.assertIsNone(ContinuousPlanner.reverse(start,q,esdf,origin,.05,robot,unknown))
        free = np.ones((100,100),bool)
        output = ContinuousPlanner.reverse(start,q,esdf,origin,.05,robot,free)
        self.assertIsNotNone(output)
        self.assertAlmostEqual(output[0][-1,0], -.24)
        esdf[41:45,45:55] = 0.
        self.assertIsNone(ContinuousPlanner.reverse(start,q,esdf,origin,.05,robot,free))

    def test_recovery_only_after_stall_and_is_bounded(self):
        from tinynav.core.continuous_planner import ContinuousPlanner
        from unittest.mock import patch
        from dataclasses import replace
        robot = replace(GO2_CONFIG, safety_radius=.08)
        q = Rotation.from_matrix(np.array([[0,0,1],[-1,0,0],[0,-1,0]])).as_quat()
        state = ContinuousPlanner()
        args = (np.zeros(3),q,np.array([1.,0.,0.]),np.full((100,100),10.),np.array([-2.5,-2.5,0.]),.05,robot)
        free = np.ones((100,100),bool)
        with patch('tinynav.core.continuous_planner.plan',return_value=(None,{'reason':'rollout_collision'})):
            self.assertIsNone(state.step(*args,0.,free)[0])
            output,info = state.step(*args,2.1,free)
            self.assertEqual(info['reason'],'recover_reverse')
            self.assertLess(output[1][0],0.)
            state.step(*args,8.,free)
            state.step(*args,14.,free)
            output,info = state.step(*args,20.,free)
            self.assertIsNone(output)
            self.assertEqual(info['reason'],'recovery_exhausted')

    def test_blocked_map_has_no_route(self):
        self.assertIsNone(guide_path(np.zeros(3), np.ones(3), np.zeros((100,100)), np.array([-2.5,-2.5,0.]), .05,.45))

if __name__ == '__main__':
    unittest.main()

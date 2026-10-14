"""Regression: unobserved occupied voxels persist until free-ray evidence."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from tinynav.core.planning_node import PlanningNode

class ObstacleMemoryTest(unittest.TestCase):
    def test_unobserved_obstacle_persists_and_free_rays_clear_it(self):
        node = SimpleNamespace(origin=np.array([-.1,-.1,-.1]), grid_shape=(4,4,4),
                               resolution=.05, step=1, planner_type='continuous',
                               occupancy_grid=np.full((4,4,4), .2))
        def update(evidence):
            with patch('tinynav.core.planning_node.run_raycasting_loopy',return_value=evidence):
                PlanningNode.update_occupancy_grid(node,np.zeros((1,1)),np.eye(4),1,1,0,0)
        for _ in range(100):
            update(np.zeros((4,4,4)))
        self.assertTrue(np.all(node.occupancy_grid == .2))
        for _ in range(3):
            update(np.full((4,4,4),-.1))
        self.assertTrue(np.all(node.occupancy_grid < 0.))

class TrajectoryContractTest(unittest.TestCase):
    def test_dense_curve_publishes_and_tracks_at_point_one_seconds(self):
        from scipy.spatial.transform import Rotation
        from std_msgs.msg import Header
        from tinynav.platforms.simulator_control import SimulatorControlNode
        paths, commands = [], []
        yaw = np.arange(31) * .05
        positions = np.zeros((31, 3))
        for i in range(1,31):
            positions[i,:2] = positions[i-1,:2] + .03*np.array([np.cos(yaw[i]),np.sin(yaw[i])])
        camera_rotation = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        quats = Rotation.from_matrix(Rotation.from_euler('z',yaw).as_matrix() @ camera_rotation).as_quat()
        trajectory = np.column_stack([positions,quats])
        publisher = SimpleNamespace(publish=paths.append)
        PlanningNode.publish_selected_path(SimpleNamespace(path_pub=publisher),np.array([trajectory]),[0],Header(),stride=1)
        self.assertEqual(len(paths[0].poses),31)
        control = SimpleNamespace(last_path=None, T_robot_to_camera=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]]),
            lookahead_steps=1, planner_dt=.1,path_pose_stride=1,max_forward_speed=1.,max_angular_speed=.75,
            cmd_pub=SimpleNamespace(publish=commands.append))
        SimulatorControlNode.path_callback(control,paths[0])
        self.assertAlmostEqual(commands[0].linear.x,.3,places=2)
        self.assertAlmostEqual(commands[0].angular.z,.5,places=5)

if __name__ == '__main__':
    unittest.main()

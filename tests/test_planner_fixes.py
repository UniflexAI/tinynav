import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from tinynav.core.planning_node import PlanningNode, generate_trajectory_library_3d, generate_predefined_trajectory_vocabularies
from tinynav.core.robot_specs import robot_config_from_dict, GO2_CONFIG
from dataclasses import asdict

class PlannerFixTests(unittest.TestCase):
    def test_time_origin_and_endpoint(self):
        q = np.array([0., 0., 0., 1.])
        p = np.array([1., 2., 3.])
        traj, params = generate_trajectory_library_3d(init_p=p, init_q=q)
        np.testing.assert_allclose(traj[:, 0, :3], np.tile(p, (len(traj), 1)))
        i = np.where((params[:, 0] == .5) & (params[:, 1] == 0))[0][0]
        # Existing ground-plane projection keeps world Z fixed.
        np.testing.assert_allclose(traj[i, -1, :3], p)
        reverse, _ = generate_predefined_trajectory_vocabularies(init_p=p, init_q=q)
        np.testing.assert_allclose(reverse[0, 0, :3], p)

    def test_horizontal_travel_duration(self):
        from scipy.spatial.transform import Rotation
        q = Rotation.from_euler('y', np.pi / 2).as_quat()
        traj, params = generate_trajectory_library_3d(init_q=q)
        i = np.where((params[:, 0] == .5) & (params[:, 1] == 0))[0][0]
        self.assertAlmostEqual(traj[i, -1, 0], 1.5)
        reverse, _ = generate_predefined_trajectory_vocabularies(init_q=q)
        self.assertAlmostEqual(reverse[0, -1, 0], -.9)

    def test_decay_factor_per_update(self):
        for factor in (0.0, 0.9, 0.99, 1.0):
            node = SimpleNamespace(origin=np.zeros(3), grid_shape=(2,2,2), resolution=1., occupancy_grid=np.full((2,2,2), .2), step=1, occupancy_decay_factor=factor)
            T = np.eye(4); T[:3,3] = 1.
            with patch('tinynav.core.planning_node.run_raycasting_loopy', return_value=np.zeros((2,2,2))):
                for count in range(1,4):
                    PlanningNode.update_occupancy_grid(node, np.zeros((1,1)), T, 1,1,0,0)
                    np.testing.assert_allclose(node.occupancy_grid, .2 * factor ** count)

    def test_published_pose_times(self):
        from std_msgs.msg import Header
        from builtin_interfaces.msg import Time
        messages = []
        node = SimpleNamespace(path_pub=SimpleNamespace(publish=messages.append))
        traj = np.zeros((1,31,7)); traj[:,:,6] = 1.
        PlanningNode.publish_selected_path(node, traj, [0], Header(stamp=Time(sec=10)))
        self.assertEqual([p.header.stamp.sec for p in messages[0].poses], [10,11,12,13])

    def test_legacy_shape_uses_rectangle(self):
        values = asdict(GO2_CONFIG)
        values.update(shape='circle', radius=5.)
        robot = robot_config_from_dict(values)
        self.assertEqual(robot.half_size, (robot.length / 2., robot.width / 2.))
        self.assertNotIn('shape', asdict(robot))
        self.assertNotIn('radius', asdict(robot))

    def test_robot_override_validation(self):
        values = asdict(GO2_CONFIG); values['safety_radius'] = .35; values['obstacle']['dilation_cells'] = 3
        result = robot_config_from_dict(values)
        self.assertEqual(result.safety_radius, .35)
        self.assertEqual(result.obstacle.dilation_cells, 3)
        values['obstacle']['dilation_cells'] = -1
        with self.assertRaises(ValueError): robot_config_from_dict(values)

if __name__ == '__main__': unittest.main()

import unittest
from unittest.mock import patch

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from scipy.spatial.transform import Rotation
from std_msgs.msg import Bool

from tinynav.platforms.cmd_vel_control import CmdVelControlNode


class CmdVelTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = CmdVelControlNode()
        self.clock = patch('tinynav.platforms.cmd_vel_control.time.monotonic', return_value=100)
        self.now = self.clock.start()
        self.node.last_cmd_pub_time = 100
        self.node._on_nav_active(Bool(data=True))
        self.set_odometry(0, 0, 0)

    def tearDown(self):
        self.clock.stop()
        self.node.destroy_node()

    def pose(self, x, y, yaw):
        body = np.eye(4)
        body[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        body[:3, 3] = [x, y, 0]
        camera = body @ np.linalg.inv(self.node.T_robot_to_camera)
        msg = PoseStamped()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, camera[:3, 3])
        q = Rotation.from_matrix(camera[:3, :3]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
        return msg

    def set_odometry(self, x, y, yaw):
        msg = Odometry()
        msg.pose.pose = self.pose(x, y, yaw).pose
        camera = self.pose(x, y, yaw).pose.orientation
        rotation = Rotation.from_quat([camera.x, camera.y, camera.z, camera.w]).as_matrix()
        position = np.array([x, y, 0]) + rotation @ self.node.robot.cam_offset_3d
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, position)
        self.node.pose_callback(msg)

    def set_path(self, reverse=False):
        path = Path()
        path.poses = [self.pose(0, 0, 0), self.pose(-0.5 if reverse else 0.5, 0, 0)]
        self.node.path_callback(path)

    def command(self):
        with patch.object(self.node.cmd_pub, 'publish') as publish:
            self.node.cmd_timer_callback()
            return publish.call_args.args[0] if publish.called else None

    def test_corrects_displacement_from_unchanged_path(self):
        self.set_path()
        self.set_odometry(0, 0.5, 0.3)
        self.now.return_value = 100.1
        cmd = self.command()
        self.assertLess(cmd.angular.z, -0.1)
        self.assertGreaterEqual(cmd.linear.x, 0)

    def test_navigation_deactivation_stops_then_yields_to_teleop(self):
        self.set_path()
        with patch.object(self.node.cmd_pub, 'publish') as publish:
            self.node._on_nav_active(Bool(data=False))
            self.assertEqual(publish.call_args.args[0].linear.x, 0)
        self.assertIsNone(self.command())

    def test_pause_clears_command_and_pid_state(self):
        self.set_path()
        self.node.tracker.linear.integral = 0.5
        self.node._on_paused(Bool(data=True))
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0, 0))
        self.assertEqual(self.node.tracker.linear.integral, 0)

    def test_stale_path_cannot_be_overridden_by_pid(self):
        self.set_path()
        self.now.return_value = 102
        self.set_odometry(0, 0.5, 0.3)
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0, 0))

    def test_stale_odometry_stops_even_with_a_fresh_path(self):
        self.now.return_value = 101
        self.set_path()
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0, 0))

    def test_reverse_remains_straight_at_fixed_speed(self):
        self.set_path(reverse=True)
        self.set_odometry(0, 0.5, 0.3)
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (-0.3, 0))

    def test_invalid_odometry_stops(self):
        self.set_path()
        invalid = Odometry()
        invalid.pose.pose.orientation.w = 0.0
        self.node.pose_callback(invalid)
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0, 0))

    def test_invalid_path_stops(self):
        self.set_path()
        invalid = Path()
        invalid.poses = [self.pose(0, 0, 0), self.pose(1, 0, 0)]
        invalid.poses[1].pose.position.x = float("nan")
        self.node.path_callback(invalid)
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0, 0))

    def test_robot_limits_hold_under_large_error(self):
        self.set_path()
        self.set_odometry(-100, -100, 0)
        self.now.return_value = 100.2
        cmd = self.command()
        self.assertLessEqual(cmd.linear.x, self.node.robot.max_linear_vel)
        self.assertLessEqual(abs(cmd.angular.z), self.node.robot.max_angular_vel)

    def test_closed_loop_with_real_node_limits_and_camera_offset(self):
        pose = np.array([0.0, 0.5, 0.3])
        dt = 0.1
        for tick in range(200):
            self.now.return_value = 100 + tick * dt
            self.set_odometry(*pose)
            path = Path()
            x = tick * dt * 0.4
            path.poses = [self.pose(x, 0, 0), self.pose(x + 0.4, 0, 0)]
            self.node.path_callback(path)
            cmd = self.command()
            pose += dt * np.array([cmd.linear.x * np.cos(pose[2]),
                                   cmd.linear.x * np.sin(pose[2]), cmd.angular.z])
        self.assertLess(abs(pose[1]), 0.1)
        self.assertLess(abs(pose[2]), 0.1)
        self.assertLess(abs(pose[0] - 8.0), 0.1)

    def test_invalid_odometry_also_stops_reverse(self):
        self.set_path(reverse=True)
        self.node.pose.pose.pose.position.x = float('nan')
        cmd = self.command()
        self.assertEqual((cmd.linear.x, cmd.angular.z), (0, 0))

    def test_camera_offset_matches_planner_control_center(self):
        self.set_odometry(0.0, 0.0, 0.0)
        self.set_path()
        self.node.max_linear_acc = 1000.0
        cmd = self.command()
        self.assertAlmostEqual(cmd.linear.x, 0.5)
        self.assertAlmostEqual(cmd.angular.z, 0.0)

    def test_pause_zero_is_delivered_through_ros(self):
        observer = rclpy.create_node('tracking_test_observer')
        received = []
        observer.create_subscription(Twist, '/cmd_vel', received.append, 10)
        try:
            self.node._on_paused(Bool(data=True))
            for _ in range(50):
                self.node.cmd_timer_callback()
                rclpy.spin_once(observer, timeout_sec=0.02)
                if received:
                    break
            self.assertTrue(received)
            self.assertEqual((received[-1].linear.x, received[-1].angular.z), (0, 0))
        finally:
            observer.destroy_node()


if __name__ == '__main__':
    unittest.main()

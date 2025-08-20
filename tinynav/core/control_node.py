import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import pygame
import sys
from math_utils import quat_to_matrix
from copy import deepcopy

class ControlMode:
    MANUAL = 0
    AUTO = 2

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.target_pose_pub = self.create_publisher(Odometry, '/control/target_pose', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)

        self.control_mode = ControlMode.MANUAL
        self.planned_velocity = Twist()
        self.get_logger().info("Starting in MANUAL control mode.")

        # Initialize pygame joystick
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            self.get_logger().error("No joystick detected!")
            sys.exit(1)
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.get_logger().info(f'Using joystick: {self.joystick.get_name()}')

        self.timer = self.create_timer(0.1, self.joystick_loop)
        self.current_pose = None
        self.last_path = None

    def path_callback(self, msg):
        self.last_path = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def joystick_loop(self):
        pygame.event.pump()

        # Mode switching with buttons
        if self.joystick.get_button(4):
            if self.control_mode == ControlMode.MANUAL:
                self.get_logger().info("Switching to Auto control mode.")
                self.control_mode = ControlMode.AUTO
            elif self.control_mode == ControlMode.AUTO:
                self.get_logger().info("Switching to Manual mode.")
                self.control_mode = ControlMode.MANUAL

        if self.control_mode == ControlMode.MANUAL:
            self.manual_control()
        elif self.control_mode == ControlMode.AUTO:
            self.auto_control()

    def manual_control(self):
        forward_speed = -self.joystick.get_axis(1)  # Left stick vertical
        angular_velocity = -self.joystick.get_axis(3)  # Right stick horizontal

        # Deadzone
        forward_speed = 0 if abs(forward_speed) < 0.1 else forward_speed
        angular_velocity = 0 if abs(angular_velocity) < 0.1 else angular_velocity

        cmd = Twist()
        cmd.linear.x = forward_speed * 0.5  # Max 0.5 m/s
        cmd.angular.z = angular_velocity * 1.0  # Max 1.0 rad/s
        self.cmd_pub.publish(cmd)

    def auto_control(self):
        # Publish the target pose from the joystick
        if self.current_pose is None:
            self.get_logger().warn("Current pose not available, cannot compute target pose.")
            return
        axis_left_forward = -self.joystick.get_axis(1)  # Left stick vertical
        axis_left_right = self.joystick.get_axis(0)  # Left stick horizontal

        # deadzone
        axis_left_forward = 0 if abs(axis_left_forward) < 0.1 else axis_left_forward
        axis_left_right = 0 if abs(axis_left_right) < 0.1 else axis_left_right

        max_speed = 0.5 * 5 # 1 m/s x 5 s
        relative_position = np.array([axis_left_right, 0.0, axis_left_forward]) * max_speed

        camera_orientation = np.array([
            self.current_pose.pose.pose.orientation.x,
            self.current_pose.pose.pose.orientation.y,
            self.current_pose.pose.pose.orientation.z,
            self.current_pose.pose.pose.orientation.w
        ])
        camera_orientation = quat_to_matrix(camera_orientation)
        relative_position = camera_orientation @ relative_position

        target_pose = deepcopy(self.current_pose)
        target_pose.pose.pose.position.x = self.current_pose.pose.pose.position.x + relative_position[0]
        target_pose.pose.pose.position.y = self.current_pose.pose.pose.position.y + relative_position[1]
        target_pose.pose.pose.position.z = self.current_pose.pose.pose.position.z + relative_position[2]
        self.target_pose_pub.publish(target_pose)

        # Publish the planned velocity
        ## calculate diff between current time, and last path time
        if self.last_path is None:
            return
        dt = 0.1  # Time step from trajectory generation
        last_path_time = self.last_path.header.stamp.sec + self.last_path.header.stamp.nanosec * 1e-9
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        time_diff = current_time - last_path_time
        idx = round(time_diff / dt)
        if idx < 0 or idx >= len(self.last_path.poses):
            self.get_logger().warn("Index out of bounds for path poses, cannot publish planned velocity.")
            return
        # get the velocity from the path.pose[idx] - path.pose[idx]
        p1 = self.last_path.poses[idx].pose.position
        p2 = self.last_path.poses[idx + 1].pose.position
        linear_velocity_vec = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]) / dt
        q1 = self.last_path.poses[idx].pose.orientation
        q2 = self.last_path.poses[idx + 1].pose.orientation
        r1 = R.from_quat([q1.x, q1.y, q1.z, q1.w])
        r2 = R.from_quat([q2.x, q2.y, q2.z, q2.w])
        angular_velocity_vec = (r2 * r1.inv()).as_rotvec() / dt
        cmd = Twist()
        cmd.linear.x = linear_velocity_vec[0] * 0.5  # Max 0.5 m/s
        cmd.angular.z = angular_velocity_vec[2] * 1.0  # Max 1.0 rad/s

        cmd.linear.x = np.clip(cmd.linear.x, -0.5, 0.5)
        cmd.angular.z = np.clip(cmd.angular.z, -1.0, 1.0)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()



import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from tinynav.core.robot_specs import ROBOT_CONFIG


class SimulatorControlNode(Node):
    def __init__(self):
        super().__init__('simulator_control_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)

        self.last_path = None
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        self.planner_dt = 0.1
        self.path_pose_stride = 10
        self.lookahead_steps = 1
        self.max_forward_speed = ROBOT_CONFIG.max_linear_vel
        self.max_angular_speed = ROBOT_CONFIG.max_angular_vel

    def path_callback(self, msg):
        self.last_path = msg
        if self.last_path is None:
            return
        if len(self.last_path.poses) < 2:
            self.get_logger().warn("Index out of bounds for path poses, cannot publish planned velocity.")
            return

        def msg2np(pose_stamped):
            T = np.eye(4)
            position = pose_stamped.pose.position
            quat = pose_stamped.pose.orientation
            T[:3, :3] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
            T[:3, 3] = np.array([position.x, position.y, position.z]).ravel()
            return T

        step_idx = int(min(self.lookahead_steps, len(self.last_path.poses) - 1))
        T1 = msg2np(self.last_path.poses[0]) @ self.T_robot_to_camera
        T2 = msg2np(self.last_path.poses[step_idx]) @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T1) @ T2
        dt = self.planner_dt * self.path_pose_stride * max(1, step_idx)
        p = T_robot_2_to_1[:3, 3]
        linear_velocity_vec = p / dt
        angular_velocity_vec = R.from_matrix(T_robot_2_to_1[:3, :3]).as_rotvec() / dt

        cmd = Twist()
        cmd.linear.x = float(np.clip(linear_velocity_vec[0], -self.max_forward_speed, self.max_forward_speed))
        cmd.angular.z = float(np.clip(angular_velocity_vec[2], -self.max_angular_speed, self.max_angular_speed))

        print(f"cmd: {cmd}")
        self.cmd_pub.publish(cmd)




def main(args=None):
    rclpy.init(args=args)
    node = SimulatorControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


class SimulatorControlNode(Node):
    def __init__(self):
        super().__init__('simulator_control_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)

        self.last_path = None

    def path_callback(self, msg):
        self.last_path = msg
        # Publish the planned velocity
        ## calculate diff between current time, and last path time
        if self.last_path is None:
            return
        dt = 0.1  # Time step from trajectory generation
        idx = 1
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
        cmd.angular.z = np.clip(cmd.angular.z, -0.5, 0.5)
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


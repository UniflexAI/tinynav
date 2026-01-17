import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from tinynav.core.math_utils import msg2np
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CmdVelControlNode(Node):
    def __init__(self):
        super().__init__('cmd_vel_control_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)
        self.T_robot_to_camera = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]]
        )
        self.last_path_time = 0.0
        self.pose = None
        self.logger = logging.getLogger(__name__)
        
    def pose_callback(self, msg):
        self.pose = msg
        
    def path_callback(self, msg):
        self.path = msg
        if self.path is None or self.pose is None:
            return
        
        #last_path_time = self.path.header.stamp.sec + self.path.header.stamp.nanosec * 1e-9
        current_time = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        
        path_time_diff = current_time - self.last_path_time
        self.logger.debug(f"diff between path and current time: {path_time_diff}")
        self.last_path_time = current_time

        T1 = msg2np(self.pose.pose)
        T2 = msg2np(self.path.poses[1])
        T_robot_1 = T1 @ self.T_robot_to_camera
        T_robot_2 = T2 @ self.T_robot_to_camera
        T_robot_2_to_1 = np.linalg.inv(T_robot_1) @ T_robot_2
        p = T_robot_2_to_1[:3, 3]
        dt = 1.0
        linear_velocity_vec = p / dt
        r = R.from_matrix(T_robot_2_to_1[:3, :3])
        angular_velocity_vec = r.as_rotvec() / dt

        vx = np.clip(linear_velocity_vec[0], -0.1, 0.5)
        vy = 0.0
        vyaw = np.clip(angular_velocity_vec[2], -0.5, 0.5)
        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = vyaw
        self.logger.debug(f"vx : {vx}, vy : {vy}, az : {vyaw}")
        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.get_logger().info("Destroying cmd_vel_control connection.")
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    node = CmdVelControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        
if __name__ == '__main__':
    main()


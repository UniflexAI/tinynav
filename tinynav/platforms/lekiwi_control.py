import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.robots.lekiwi.lekiwi import LeKiwi
import math
from scipy.spatial.transform import Rotation as R
import numpy as np

class LeKiwiControlNode(Node):
    def __init__(self):
        super().__init__('lekiwi_control')

        self.get_logger().info("Configuring LeKiwi.")
        self.robot_config = LeKiwiConfig()
        self.robot = LeKiwi(self.robot_config)
        self.robot.connect()
        
        self.cmd_pub = self.create_publisher(Twist, '/lekiwi_control/cmd_vel', 10)
        self.create_subscription(Path, '/planning/trajectory_path', self.path_callback, 10)

        self.last_path = None

    def _send_lekiwi_cmd(self, msg):
        data = {
            "x.vel": 0,
            "y.vel": 0,
            "theta.vel": 0,
        }

        data["x.vel"] = msg.linear.x 
        data["theta.vel"] = msg.angular.z * 180 / math.pi
        _action_sent = self.robot.send_action(data)
        self.get_logger().info(f"Speed: {msg.linear.x}m/s, Angular: {msg.angular.z}rad/s")

    def path_callback(self, msg):
        self.last_path = msg
        # Publish the planned velocity
        ## calculate diff between current time, and last path time
        if self.last_path is None:
            return
        dt = 0.1  # Time step from trajectory generation
        # find the index of the path that is closest to the current time 
        last_path_time = self.last_path.header.stamp.sec + self.last_path.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().to_msg()
        current_time = now.sec + now.nanosec * 1e-9
        time_diff = current_time - last_path_time
        idx = round(time_diff / dt)
        if idx < 0 or idx >= len(self.last_path.poses)-1:
            self.cmd_pub.publish(Twist()) # Stop the robot
            self._send_lekiwi_cmd(Twist())
            self.get_logger().warn("Index out of bounds for path poses, cannot publish planned velocity.")
            return
        # get the velocity from the path.pose[idx] - path.pose[idx]
        p1 = self.last_path.poses[idx].pose.position
        p2 = self.last_path.poses[idx + 1].pose.position
        q1 = self.last_path.poses[idx].pose.orientation
        q2 = self.last_path.poses[idx + 1].pose.orientation
        r1 = R.from_quat([q1.x, q1.y, q1.z, q1.w])
        r2 = R.from_quat([q2.x, q2.y, q2.z, q2.w])
        linear_velocity_vec = r1.inv().apply(np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])/ dt) 
        angular_velocity_vec = (r2.inv() * r1).as_rotvec() / dt
        
        cmd = Twist()
        cmd.linear.x = np.clip(linear_velocity_vec[2], -2.0, 2.0)
        cmd.angular.z = np.clip(angular_velocity_vec[1], -1.0, 1.0)
        self._send_lekiwi_cmd(cmd)
        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        self.get_logger().info("Destroying LeKiwi connection.")
        self.robot.stop_base()
        self.robot.disconnect()
        super().destroy_node() 
        
def main(args=None):
    rclpy.init(args=args)
    node = LeKiwiControlNode()
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

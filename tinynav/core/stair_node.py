"""Stair mode node: replaces map_node as the source of /control/target_pose inside a stairwell.

Start with --direction up|down, or send "up" / "down" on /stair/cmd; "stop" leaves stair mode.
Add the U-turn side at landings if known ("up left", or --turn left); otherwise it is estimated. No map or relocalization
is used; targets come from local geometry (see stair_target.py) at a low rate, like map_node.
When there is no safe target (odometry jump, no ground under the feet) the planning target is
cleared through /mapping/poi_change, so planning stops publishing paths and cmd_vel stops.
"""
import argparse

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from tinynav.core.math_utils import msg2np, np2msg
from tinynav.core.stair_target import StairConfig, StairTargetGenerator

STOP_STATUSES = ('no_seed', 'odom_invalid')
TURN_SIDES = {'auto': 0, 'left': 1, 'right': -1}


def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class StairNode(Node):
    def __init__(self, args):
        super().__init__('stair_node')
        self.gen = StairTargetGenerator(StairConfig(camera_height=args.camera_height))
        self.gen.turn_side = TURN_SIDES[args.turn]
        self.bridge = CvBridge()
        self.K = None
        self.direction = args.direction  # None: stair mode inactive
        self.latest_T = None
        self.stopped = True  # planning target is currently cleared by us

        self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.create_subscription(Odometry, '/slam/odometry', self.odom_callback, 100)
        self.create_subscription(String, '/stair/cmd', self.cmd_callback, 10)
        depth_sub = message_filters.Subscriber(self, Image, '/slam/depth')
        pose_sub = message_filters.Subscriber(self, Odometry, '/slam/odometry_visual')
        self.ts = message_filters.TimeSynchronizer([depth_sub, pose_sub], queue_size=10)
        self.ts.registerCallback(self.depth_callback)

        self.target_pub = self.create_publisher(Odometry, '/control/target_pose', 10)
        self.poi_change_pub = self.create_publisher(Odometry, '/mapping/poi_change', 10)
        self.status_pub = self.create_publisher(String, '/stair/status', 10)
        self.path_pub = self.create_publisher(Path, '/stair/path', 10)
        self.create_timer(1.0 / args.rate, self.timer_callback)
        self.get_logger().info(f'stair_node ready, direction={self.direction}; send "up" / "down" / "stop" on /stair/cmd')

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def cmd_callback(self, msg):
        words = msg.data.strip().lower().split()
        if words and words[0] in ('up', 'down') and (len(words) == 1 or (len(words) == 2 and words[1] in TURN_SIDES)):
            self.gen.new_run()  # keeps the recent height map
            self.gen.turn_side = TURN_SIDES[words[1] if len(words) == 2 else 'auto']
            self.direction = words[0]
            self.get_logger().info(f'stair mode on, going {self.direction}, turn {words[1] if len(words) == 2 else "auto"}')
        elif words == ['stop']:
            self.direction = None
            self.stop_robot('stair mode off')
        else:
            self.get_logger().warning(f'unknown /stair/cmd {msg.data!r}, expected "up|down [left|right|auto]" or "stop"')

    def odom_callback(self, msg):
        was_valid = self.gen.odom_valid
        p = msg.pose.pose.position
        self.gen.add_pose(stamp_sec(msg.header.stamp), (p.x, p.y, p.z))
        if self.direction is not None and was_valid and not self.gen.odom_valid:
            # do not wait for the next timer tick
            self.stop_robot('odometry jump')
            self.publish_status('odom_invalid')

    def depth_callback(self, depth_msg, odom_msg):
        if self.K is None:
            return
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        T, _ = msg2np(odom_msg)
        self.gen.add_depth(stamp_sec(odom_msg.header.stamp), depth, self.K, T)
        self.latest_T = T

    def timer_callback(self):
        if self.direction is None or self.latest_T is None:
            return
        res = self.gen.compute(self.latest_T, self.direction)
        self.publish_status(res['status'])
        if res['status'] in STOP_STATUSES or res['target'] is None:
            self.stop_robot(res['status'])
            return
        now = self.get_clock().now().to_msg()
        target = np.eye(4)
        target[:3, 3] = res['target']
        self.target_pub.publish(np2msg(target, now, 'world', 'camera'))
        self.stopped = False
        if res['path'] is not None:
            path = Path()
            path.header.stamp = now
            path.header.frame_id = 'world'
            for x, y, z in res['path']:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(x), float(y), float(z)
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.path_pub.publish(path)

    def publish_status(self, status):
        self.status_pub.publish(String(data=f'{self.direction} {status} well_side={self.gen.well_side:+.1f}'))

    def stop_robot(self, reason):
        if self.stopped:
            return
        # planning drops its target on poi_change, stops publishing paths, and cmd_vel times out to zero
        self.poi_change_pub.publish(np2msg(np.eye(4), self.get_clock().now().to_msg(), 'world', 'map'))
        self.stopped = True
        self.get_logger().warning(f'stop: {reason}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=float, default=2.0, help='target update rate in Hz (map_node uses 2 Hz)')
    parser.add_argument('--camera_height', type=float, default=0.66, help='camera height above the ground [m]')
    parser.add_argument('--direction', choices=['up', 'down'], default=None, help='start in stair mode right away')
    parser.add_argument('--turn', choices=list(TURN_SIDES), default='auto', help='U-turn side at landings (auto: estimate)')
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = StairNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

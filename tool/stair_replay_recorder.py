"""Record planning + stair_node outputs during a rosbag replay (see scripts/run_stair_replay_test.sh).

Topics stamped with wall-clock time (target, stair path/status, stop) are tagged with the latest
/slam/odometry_visual stamp so everything lines up with the bag. Saved as a pickle on Ctrl+C.
"""
import pickle
import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def path_xyz(msg):
    return np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses])


class StairReplayRecorder(Node):
    def __init__(self):
        super().__init__('stair_replay_recorder')
        self.t = None
        self.data = {k: [] for k in ('plan', 'mask', 'target', 'stair_path', 'status', 'stop')}
        self.create_subscription(Odometry, '/slam/odometry_visual', self.odom_callback, 50)
        self.create_subscription(Path, '/planning/trajectory_path', lambda m: self.data['plan'].append((stamp(m.header), path_xyz(m))), 50)
        self.create_subscription(OccupancyGrid, '/planning/obstacle_mask', self.mask_callback, 50)
        self.create_subscription(Odometry, '/control/target_pose', lambda m: self.tag('target', np.array(
            [m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z])), 50)
        self.create_subscription(Path, '/stair/path', lambda m: self.tag('stair_path', path_xyz(m)), 50)
        self.create_subscription(String, '/stair/status', lambda m: self.tag('status', m.data), 50)
        self.create_subscription(Odometry, '/mapping/poi_change', lambda m: self.tag('stop', None), 50)

    def odom_callback(self, msg):
        self.t = stamp(msg.header)

    def tag(self, key, value):
        if self.t is not None:
            self.data[key].append((self.t, value))

    def mask_callback(self, msg):
        mask = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width), order='F') > 50
        origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
        self.data['mask'].append((stamp(msg.header), origin, msg.info.resolution, np.packbits(mask), mask.shape))


def main():
    rclpy.init()
    node = StairReplayRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    with open(sys.argv[1], 'wb') as f:
        pickle.dump(node.data, f)
    print({k: len(v) for k, v in node.data.items()})


if __name__ == '__main__':
    main()

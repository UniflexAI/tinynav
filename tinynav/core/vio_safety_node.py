"""Persistent VIO supervision; no robot SDK or actuator is loaded here."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

from tinynav.core.vio_safety import VioSafety


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class VioSafetyNode(Node):
    def __init__(self, state_path=None, source_clock=None):
        super().__init__('vio_safety')
        self.owner = None
        if state_path:
            Path(state_path).parent.mkdir(parents=True, exist_ok=True)
            self.owner = open(str(state_path) + '.owner.lock', 'a')
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.guard = VioSafety(state_path)
        self.source_clock = source_clock or (lambda: self.get_clock().now().nanoseconds / 1e9)
        self.bad_source = set()
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '/nav/vio_safety', qos)
        self.create_subscription(Odometry, '/slam/odometry', self.pose_callback, 100)
        self.create_subscription(Image, '/slam/depth', self.depth_callback, 10)
        self.create_subscription(String, '/camera/camera/vio_status', self.status_callback, 10)
        self.declare_parameter('confirm_fault_id', 0)
        self.add_on_set_parameters_callback(self.confirm_parameters)
        self.create_timer(.1, self.publish_state)

    def pose_callback(self, msg):
        if not self.source_fresh(seconds(msg.header.stamp), 'pose'):
            return
        p = msg.pose.pose.position
        self.guard.pose(seconds(msg.header.stamp), (p.x, p.y, p.z), time.monotonic())
        self.publish_state()

    def depth_callback(self, msg):
        if not self.source_fresh(seconds(msg.header.stamp), 'depth'):
            return
        if not msg.width or not msg.height or len(msg.data) < msg.height * msg.step:
            return
        if msg.encoding == '32FC1':
            sample = np.frombuffer(msg.data, np.float32)[::1000]
            if not len(sample) or np.mean(np.isfinite(sample) & (sample > 0.)) < .05:
                return
        self.guard.depth(seconds(msg.header.stamp), time.monotonic())

    def source_fresh(self, stamp, stream):
        age = self.source_clock() - stamp
        if not np.isfinite(age) or age > 1. or age < -.1:
            if stream not in self.bad_source:
                self.guard.fault(stream + '_stamp_invalid', time.monotonic())
                self.bad_source.add(stream)
                self.publish_state()
            return False
        self.bad_source.discard(stream)
        return True

    def status_callback(self, msg):
        self.guard.status(msg.data, time.monotonic())
        self.publish_state()

    def publish_state(self):
        state = self.guard.snapshot(time.monotonic())
        self.state_pub.publish(String(data=json.dumps(state)))
        return state

    def confirm_parameters(self, params):
        values = [p.value for p in params if p.name == 'confirm_fault_id']
        if not values:
            return SetParametersResult(successful=True)
        if len(values) != 1 or type(values[0]) is not int:
            return SetParametersResult(successful=False, reason='Expected integer fault ID')
        ok, reason = self.guard.confirm(values[0], time.monotonic())
        self.publish_state()
        return SetParametersResult(successful=ok, reason=reason)

    def destroy_node(self):
        if self.owner:
            self.owner.close()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-file', default=os.path.join(os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db'), 'vio_safety.json'))
    args = parser.parse_args()
    rclpy.init()
    node = VioSafetyNode(args.state_file)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

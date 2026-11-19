"""Measure breakaway and sustained motion. Run locally on a stationary robot."""
import argparse
from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import time

import rclpy
import rosbag2_py as bag
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String


class Trial(Node):
    def __init__(self, path):
        super().__init__('deadzone_calibration')
        self.writer = bag.SequentialWriter()
        self.writer.open(bag.StorageOptions(uri=str(path), storage_id='sqlite3'),
                         bag.ConverterOptions('', ''))
        self.history = deque(maxlen=200)
        self.active = False
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        for topic, msg, typ in [('/cmd_vel', Twist, 'geometry_msgs/msg/Twist'),
                                ('/slam/odometry_visual', Odometry, 'nav_msgs/msg/Odometry'),
                                ('/slam/odometry', Odometry, 'nav_msgs/msg/Odometry'),
                                ('/nav/active', Bool, 'std_msgs/msg/Bool'),
                                ('/calibration/trial', String, 'std_msgs/msg/String')]:
            self.writer.create_topic(bag.TopicMetadata(name=topic, type=typ, serialization_format='cdr'))
            if topic != '/calibration/trial':
                qos = (QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
                       if topic == '/nav/active' else qos_profile_sensor_data)
                self.create_subscription(msg, topic, lambda m, t=topic: self.receive(t, m), qos)

    def write(self, topic, m):
        self.writer.write(topic, serialize_message(m), self.get_clock().now().nanoseconds)

    def receive(self, topic, m):
        self.write(topic, m)
        if topic == '/nav/active':
            self.active = m.data
        elif topic == '/slam/odometry_visual':
            p = m.pose.pose
            xyz = (p.position.x, p.position.y, p.position.z)
            q = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if all(map(math.isfinite, xyz + q)) and (not self.history or stamp > self.history[-1][1]):
                self.history.append((time.monotonic(), stamp, xyz, q))

    def mark(self, phase, **extra):
        self.write('/calibration/trial', String(data=json.dumps(dict(self.meta, phase=phase, **extra))))

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=min(.02, max(0., end-time.monotonic())))

    def movement(self):
        last = self.history[-1]
        first = min(self.history, key=lambda item: abs(item[0] - (last[0] - .6)))
        if last[0] - first[0] < .45:
            raise RuntimeError('Need a continuous odometry window')
        dot = abs(sum(a*b for a,b in zip(last[3], first[3])))
        norm = math.hypot(*last[3]) * math.hypot(*first[3])
        return math.dist(first[2], last[2]), 2*math.acos(min(1., dot/norm))

    def segment(self, phase, value, seconds):
        self.mark(phase, magnitude=value)
        cmd = Twist()
        if self.turn:
            cmd.angular.z = self.sign * value
        else:
            cmd.linear.x = self.sign * value
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.active or not self.history or time.monotonic()-self.history[-1][0] > .4:
                raise RuntimeError('Navigation is active or visual odometry is stale')
            self.cmd.publish(cmd)
            self.spin_for(min(1/12, max(0., end-time.monotonic())))

    def run(self, repeats):
        for direction, self.sign, self.turn in [('forward',1,False), ('reverse',-1,False),
                                                ('left',1,True), ('right',-1,True)]:
            levels = (.05,.10,.15,.20,.25) if not self.turn else (.05,.10,.15,.20,.30)
            for repeat in range(1, repeats+1):
                for mode in ('start','sustain'):
                    input(f'{direction} / {mode} / round {repeat}: position robot in clear space, stop teleop, press Enter ')
                    self.history.clear()
                    self.spin_for(1.)
                    for level in levels:
                        self.meta = dict(direction=direction, mode=mode, repeat=repeat, level=level)
                        self.segment('settle', 0., 1.5)
                        distance, angle = self.movement()
                        if distance > .025 or angle > .06:
                            raise RuntimeError('Robot still moving during stationary baseline')
                        if mode == 'sustain':
                            self.segment('lead_in', .45 if self.turn else .30, 1.5)
                            distance, angle = self.movement()
                            if (angle < .10 if self.turn else distance < .05):
                                self.mark('invalid_lead_in', distance=distance, angle=angle)
                                self.segment('stop', 0., .6)
                                continue
                        self.segment('probe', level, 2.)
                        self.segment('stop', 0., .6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tinynav/tinynav_temp') /
                        datetime.now().strftime('tinynav-deadzone-%Y%m%d-%H%M%S'))
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a new output directory and at least one repeat')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Trial(args.output)
    print(f'Bag: {args.output}', flush=True)
    try:
        node.run(args.repeats)
    finally:
        # Publish stop even after Ctrl+C. Use a ROS context that stays alive on SIGINT.
        for _ in range(6):
            node.cmd.publish(Twist())
            node.spin_for(.08)
        del node.writer
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""Right-turn calibration with time-aligned front camera, odometry and commands."""
import argparse
from datetime import datetime
import math
from pathlib import Path
import time

import rclpy
import rosbag2_py
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rosidl_runtime_py.utilities import get_message

from calibrate_deadzone import Trial
from record_nav_comparison import front_camera


class RightTurn(Trial):
    levels = (.10, .20, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75)

    def camera(self, requested):
        self.spin_for(1.)
        topic = front_camera(self, requested)
        typ = next(t for t in dict(self.get_topic_names_and_types())[topic]
                   if t in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'))
        self.writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=typ, serialization_format='cdr'))
        self.frame_time = 0.
        def frame(message):
            self.write(topic, message)
            self.frame_time = time.monotonic()
        self.create_subscription(get_message(typ), topic, frame, qos_profile_sensor_data)
        print(f'Recording front camera: {topic}', flush=True)
        self.spin_for(1.)
        self.check_camera()

    def check_camera(self):
        if time.monotonic() - self.frame_time > 1.:
            raise RuntimeError('No recent front camera frame')

    def spin_for(self, seconds):
        super().spin_for(seconds)
        if hasattr(self, 'frame_time') and self.frame_time:
            self.check_camera()

    def right_rate(self):
        # Visual odometry uses camera optical +Z as body forward.
        samples = [s for s in self.history if s[0] >= self.history[-1][0] - .65]
        if len(samples) < 2 or samples[-1][1] - samples[0][1] < .4:
            raise RuntimeError('Insufficient fresh odometry for yaw measurement')
        def yaw(sample):
            x, y, z, w = sample[3]
            return math.atan2(2*(y*z-w*x), 2*(x*z+w*y))
        return -math.remainder(yaw(samples[-1])-yaw(samples[0]), 2*math.pi)/(samples[-1][1]-samples[0][1])

    def run(self, repeats):
        self.turn, self.sign = True, -1
        for repeat in range(1, repeats+1):
            lead = None
            for mode in ('start', 'sustain'):
                if mode == 'sustain' and lead is None:
                    self.mark('no_verified_lead_in')
                    print('No confirmed right turn this round; skipping sustain.', flush=True)
                    continue
                input(f'Round {repeat}, {mode}: clear surroundings and press Enter ')
                self.history.clear()
                self.spin_for(1.)
                for level in self.levels:
                    if mode == 'sustain' and level > lead:
                        break
                    self.meta = dict(direction='right', repeat=repeat, mode=mode, level=level)
                    self.segment('settle', 0., 1.5)
                    distance, angle = self.movement()
                    if distance > .025 or angle > .06:
                        raise RuntimeError('Robot still moving during stationary baseline')
                    if mode == 'sustain':
                        self.segment('lead_in', lead, 2.)
                        if self.right_rate() < .08:
                            self.mark('invalid_lead_in')
                            self.segment('stop', 0., .6)
                            continue
                    self.segment('probe', level, 2.)
                    rate = self.right_rate()
                    self.mark('measured', right_rad_s=rate)
                    self.segment('stop', 0., .6)
                    if mode == 'start' and rate >= .08:
                        lead = level


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-topic')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', type=Path, default=Path('/tinynav/tinynav_temp') /
                   datetime.now().strftime('tinynav-right-turn-%Y%m%d-%H%M%S'))
    args = p.parse_args()
    if args.output.exists() or args.repeats < 1:
        p.error('Use a new output directory and at least one repeat')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    n = RightTurn(args.output)
    print(f'Bag: {args.output}', flush=True)
    try:
        n.camera(args.image_topic)
        n.run(args.repeats)
    finally:
        for _ in range(6):
            n.cmd.publish(Twist())
            Trial.spin_for(n, .08)
        del n.writer
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

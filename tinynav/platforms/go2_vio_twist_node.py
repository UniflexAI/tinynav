"""Looper VIO pose -> Go2 body-frame velocity (/go2/vio_twist, 20 Hz), a robot_localization twist input.

VIO is withheld for 1 s after a gap, a non-tracking status or a single-frame position jump, while its 1 s mean velocity
disagrees with the legs (/go2/leg_twist) by more than gate_mps, and when one 20 Hz sample is more than spike_mps off the
latest leg velocity: on 10-09 the stair-descent VIO drifted to 1-6.7 m/s while still reporting TRACKING, and
robot_localization's own Mahalanobis gate let most of it through; on 10-10 in a dark stairwell it jumped 0.36-1.12 m
in one frame and the 1 s mean let ~0.2 m in before it tripped. The jump test needs no legs, so it still works when
lowstate is gone. Angular rates are published too, so the EKF keeps its attitude without the dog IMU.
"""
import collections

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistWithCovarianceStamped
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import String

GOOD_STATUS = ('TRACKING', 'TRACKING_STATIC')


class Go2VioTwistNode(Node):
    def __init__(self):
        super().__init__('go2_vio_twist')
        # camera -> body, fitted from flight data (go2-odom-data make_ekf_inputs.py); dog 1, 10-08
        self.R_bc = Rot.from_euler('xyz', self.declare_parameter('cam_rpy_deg', [-90.37, 0.01, -88.81]).value, degrees=True)
        self.lever = np.array(self.declare_parameter('lever_arm_m', [0.288, 0.01, 0.077]).value)
        self.std = self.declare_parameter('std_mps', 0.03).value
        self.gate = self.declare_parameter('gate_mps', 0.4).value
        self.spike = self.declare_parameter('spike_mps', 1.0).value
        self.w_std = self.declare_parameter('rate_std_rps', 0.05).value   # above the dog gyro's 0.02, so the IMU leads
        self.span = 0.1                                    # s of VIO poses per velocity
        self.poses = collections.deque(maxlen=200)         # (t, p, R_wc), 100 Hz
        self.vio_v = collections.deque(maxlen=20)          # (t, v_body), last 1 s
        self.leg_v = collections.deque(maxlen=50)
        self.hold_until = 0.0
        self.last_tc = 0.0
        self.status = ''
        self.pub = self.create_publisher(TwistWithCovarianceStamped, '/go2/vio_twist', 20)
        self.create_subscription(PoseStamped, '/camera/camera/vio_100hz', self.on_pose, 50)
        self.create_subscription(String, '/camera/camera/vio_status', self.on_status, 10)
        self.create_subscription(TwistWithCovarianceStamped, '/go2/leg_twist', self.on_leg, 50)
        self.create_timer(0.05, self.on_timer)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def hold(self, why):
        if self.now() >= self.hold_until:
            self.get_logger().info(f'VIO withheld: {why}')
        self.hold_until = self.now() + 1.0

    def on_status(self, msg):
        self.status = msg.data
        if msg.data not in GOOD_STATUS:
            self.hold(f'status {msg.data}')

    def on_pose(self, msg):
        t = self.now()   # receive time, same clock as /go2/leg_twist (offset to lowstate measured 0 +- 10 ms)
        if self.poses and t - self.poses[-1][0] > 0.2:
            self.hold(f'gap {t - self.poses[-1][0]:.2f} s')
        p, o = msg.pose.position, msg.pose.orientation
        pos, tc = np.array([p.x, p.y, p.z]), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.poses:
            # normal frames move <= 0.04 m (p99.9 over 3 runs); jumps were 0.24-1.12 m. Camera stamps, not receive time
            step, dtc = np.linalg.norm(pos - self.poses[-1][1]), max(tc - self.last_tc, 0.0)
            if step > 0.15 + 2.0 * dtc:
                self.hold(f'jump {step:.2f} m in {dtc * 1e3:.0f} ms')
        self.last_tc = tc
        self.poses.append((t, pos, Rot.from_quat([o.x, o.y, o.z, o.w])))

    def on_leg(self, msg):
        v = msg.twist.twist.linear
        self.leg_v.append((self.now(), np.array([v.x, v.y, v.z])))

    def on_timer(self):
        if len(self.poses) < 2 or self.now() - self.poses[-1][0] > 0.2:
            return
        t1, p1, R1 = self.poses[-1]
        t0, p0, R0 = min(self.poses, key=lambda x: abs(x[0] - (t1 - self.span)))
        if t1 - t0 < 0.5 * self.span:
            return
        dt = t1 - t0
        R_wb = R1 * self.R_bc.inv()
        w_b = self.R_bc.apply((R0.inv() * R1).as_rotvec() / dt)
        v = R_wb.inv().apply((p1 - p0) / dt) - np.cross(w_b, self.lever)
        tm = 0.5 * (t0 + t1)
        self.vio_v.append((tm, v))
        legs = [x for tt, x in self.leg_v if tt > tm - 1.0]
        if len(legs) >= 5 and len(self.vio_v) > 10:   # leg twist is 10 Hz
            dv = np.linalg.norm(np.mean([x for _, x in self.vio_v], 0) - np.mean(legs, 0))
            if dv > self.gate:
                self.hold(f'disagrees with legs by {dv:.2f} m/s')
        if legs and np.linalg.norm(v - legs[-1]) > self.spike:
            self.hold(f'sample {np.linalg.norm(v - legs[-1]):.2f} m/s off the legs')
        if self.now() < self.hold_until:
            return
        m = TwistWithCovarianceStamped()
        m.header.stamp = rclpy.time.Time(seconds=tm).to_msg()
        m.header.frame_id = 'base_link'
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = map(float, v)
        m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = map(float, w_b)
        c = [0.0] * 36
        c[0] = c[7] = c[14] = self.std ** 2
        c[21] = c[28] = c[35] = self.w_std ** 2
        m.twist.covariance = c
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = Go2VioTwistNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

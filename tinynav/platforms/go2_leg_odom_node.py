"""Go2 leg odometry from rt/lowstate: stance-leg kinematics + body IMU -> body-frame velocity.

Publishes /go2/leg_twist (50 Hz, a robot_localization twist input) and, decimated to ~200 Hz for recording and
offline replay, /go2/imu and /go2/joint_states (12 joints, plus 4 *_foot entries whose effort is the foot force).
"""
import argparse
import collections
import threading

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import TwistWithCovarianceStamped
from sensor_msgs.msg import Imu, JointState

# Go2 URDF; leg order FR, FL, RR, RL (Unitree motor order)
HIP_X, HIP_Y, L1, L2, L3 = 0.1934, 0.0465, 0.0955, 0.213, 0.213
SIDE = np.array([-1.0, 1.0, -1.0, 1.0])
FRONT = np.array([1.0, 1.0, -1.0, -1.0])
LEGS = ('FR', 'FL', 'RR', 'RL')
JOINT_NAMES = [f'{leg}_{j}_joint' for leg in LEGS for j in ('hip', 'thigh', 'calf')] + [f'{leg}_foot' for leg in LEGS]

# Stance weights. FL foot force reads ~15 whatever the load on both dogs measured, so FL uses knee torque
# (>9 N*m agreed with foot-force stance 86% on the good legs); on 10-09 data this took descent height 0.47 -> 0.91.
FF_FLOOR = 30.0
KNEE_LEGS = (1,)
TAU_THR, TAU_GAIN = 9.0, 10.0
# Velocity std by terrain (m/s), measured against VIO with the same 0.1 s averaging; descent legs under-count ~30%
LEG_STD = {'flat': (0.12, 0.08, 0.05), 'up': (0.18, 0.12, 0.08), 'down': (0.6, 0.6, 0.6)}
PITCH_STAIRS = np.radians(12.0)           # body pitch > +12 deg (1 s mean) = going down
AVG_N, PUB_EVERY, PITCH_N = 50, 10, 500   # in 500 Hz lowstate samples: 0.1 s mean, 50 Hz output, 1 s pitch mean
RAW_HZ, STATE_HZ = 200, 500


def foot_pos(q):
    """Foot positions in the body frame, q: (4, 3) abad/hip/knee -> (4, 3)."""
    a, h, k = q[:, 0], q[:, 1], q[:, 2]
    x = -L2 * np.sin(h) - L3 * np.sin(h + k)
    zl = -L2 * np.cos(h) - L3 * np.cos(h + k)
    y0 = SIDE * L1
    return np.stack([x + FRONT * HIP_X, y0 * np.cos(a) - zl * np.sin(a) + SIDE * HIP_Y, y0 * np.sin(a) + zl * np.cos(a)], 1)


def body_velocity(q, dq, gyro, foot_force, tau_knee, eps=1e-4):
    """Body-frame velocity from one lowstate sample: weighted mean over stance legs of -(J dq + w x p); 0 if none."""
    p = foot_pos(q)
    v_legs = -((foot_pos(q + eps * dq) - p) / eps + np.cross(gyro, p))
    w = np.clip(foot_force - FF_FLOOR, 0.0, None)
    for leg in KNEE_LEGS:
        w[leg] = TAU_GAIN * max(abs(tau_knee[leg]) - TAU_THR, 0.0)
    s = w.sum()
    return v_legs.T @ w / s if s > 0 else np.zeros(3)


def rpy_to_quat(r, p, y):
    """Extrinsic xyz (Unitree imu_state.rpy) -> x, y, z, w."""
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class Go2LegOdomNode(Node):
    def __init__(self, network_interface):
        super().__init__('go2_leg_odom')
        self.pub_twist = self.create_publisher(TwistWithCovarianceStamped, '/go2/leg_twist', 50)
        self.pub_imu = self.create_publisher(Imu, '/go2/imu', 200)
        self.pub_joints = self.create_publisher(JointState, '/go2/joint_states', 200)
        self.vel = collections.deque(maxlen=AVG_N)
        self.pitch = collections.deque(maxlen=PITCH_N)
        self.n = 0
        self.lock = threading.Lock()
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, network_interface)
        self.sub = ChannelSubscriber('rt/lowstate', LowState_)
        self.sub.Init(self.on_lowstate, 50)
        self.get_logger().info(f'go2_leg_odom on {network_interface}')

    def on_lowstate(self, m):
        now = self.get_clock().now()
        ms = m.motor_state[:12]
        q = np.array([x.q for x in ms]).reshape(4, 3)
        dq = np.array([x.dq for x in ms]).reshape(4, 3)
        tau = np.array([x.tau_est for x in ms])
        gyro = np.array(m.imu_state.gyroscope, float)
        ff = np.array(m.foot_force, float)
        with self.lock:
            self.n += 1
            n = self.n
            self.vel.append(body_velocity(q, dq, gyro, ff, tau.reshape(4, 3)[:, 2]))
            self.pitch.append(m.imu_state.rpy[1])
            v = np.mean(self.vel, 0) if n % PUB_EVERY == 0 else None
            pitch = np.mean(self.pitch)
        if v is not None:
            cls = 'down' if pitch > PITCH_STAIRS else 'up' if pitch < -PITCH_STAIRS else 'flat'
            t = TwistWithCovarianceStamped()
            t.header.stamp = (now - Duration(nanoseconds=int(AVG_N / 2 / STATE_HZ * 1e9))).to_msg()  # window centre
            t.header.frame_id = 'base_link'
            t.twist.twist.linear.x, t.twist.twist.linear.y, t.twist.twist.linear.z = map(float, v)
            c = [0.0] * 36
            c[0], c[7], c[14] = (s * s for s in LEG_STD[cls])
            c[21] = c[28] = c[35] = 1e6   # angular part unused
            t.twist.covariance = c
            self.pub_twist.publish(t)
        if n * RAW_HZ // STATE_HZ != (n - 1) * RAW_HZ // STATE_HZ:
            stamp = now.to_msg()
            imu = Imu()
            imu.header.stamp, imu.header.frame_id = stamp, 'base_link'
            o = imu.orientation
            o.x, o.y, o.z, o.w = map(float, rpy_to_quat(*m.imu_state.rpy))
            imu.orientation_covariance = [1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 4e-4]
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, gyro)
            imu.angular_velocity_covariance = [4e-4, 0.0, 0.0, 0.0, 4e-4, 0.0, 0.0, 0.0, 4e-4]   # static 0.006, gait shocks
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, m.imu_state.accelerometer)
            self.pub_imu.publish(imu)
            js = JointState()
            js.header.stamp = stamp
            js.name = JOINT_NAMES
            js.position = q.ravel().tolist() + [0.0] * 4
            js.velocity = dq.ravel().tolist() + [0.0] * 4
            js.effort = tau.tolist() + ff.tolist()
            self.pub_joints.publish(js)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-interface', default='enP8p1s0', help='Network interface connected to the robot')
    parsed_args, ros_args = parser.parse_known_args(args=args)
    rclpy.init(args=ros_args)
    node = Go2LegOdomNode(parsed_args.network_interface)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

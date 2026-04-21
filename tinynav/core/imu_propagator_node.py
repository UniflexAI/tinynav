import logging

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu

from tinynav.core.math_utils import msg2np, np2msg


def integrate(
    odom_prev,
    imu,
    bias_acc=None,
    bias_gyro=None,
    gravity_world=np.array([0.0, 0.0, -9.80]),
):
    t0, odom_msg = odom_prev
    pose_prev, velocity_prev = msg2np(odom_msg)
    rotation_prev = pose_prev[:3, :3]
    translation_prev = pose_prev[:3, 3]

    t1, imu_msg = imu
    gyro = np.array([
        imu_msg.angular_velocity.x,
        imu_msg.angular_velocity.y,
        imu_msg.angular_velocity.z,
    ], dtype=float)
    accel = np.array([
        imu_msg.linear_acceleration.x,
        imu_msg.linear_acceleration.y,
        imu_msg.linear_acceleration.z,
    ], dtype=float)
    if bias_gyro is not None:
        gyro = gyro - bias_gyro
    if bias_acc is not None:
        accel = accel - bias_acc

    dt = t1 - t0
    delta_rotation = R.from_rotvec(gyro * dt).as_matrix()
    rotation_new = rotation_prev @ delta_rotation
    accel_world = rotation_prev @ accel + gravity_world
    velocity_new = velocity_prev + accel_world * dt
    translation_new = translation_prev + velocity_prev * dt + 0.5 * accel_world * dt * dt

    pose_new = np.eye(4)
    pose_new[:3, :3] = rotation_new
    pose_new[:3, 3] = translation_new

    odom_new = np2msg(pose_new, imu_msg.header.stamp, odom_msg.header.frame_id, odom_msg.child_frame_id, velocity_new)
    odom_new.twist.twist.angular.x = gyro[0]
    odom_new.twist.twist.angular.y = gyro[1]
    odom_new.twist.twist.angular.z = gyro[2]
    return t1, odom_new


class ImuPropagatorNode(Node):
    def __init__(self):
        super().__init__("imu_propagator_node")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1000)
        self.imu_sub = self.create_subscription(Imu, "/camera/camera/imu", self.imu_callback, qos_profile)
        self.odom_sub = self.create_subscription(Odometry, "/slam/odometry", self.odom_callback, qos_profile)
        self.bias_acc_sub = self.create_subscription(Vector3Stamped, "/slam/imu_bias_accel", self.bias_acc_callback, qos_profile)
        self.bias_gyro_sub = self.create_subscription(Vector3Stamped, "/slam/imu_bias_gyro", self.bias_gyro_callback, qos_profile)
        self.odom_pub = self.create_publisher(Odometry, "/slam/odometry_100hz", 50)

        self.imu_buffer = []
        self.odom_10hz_buffer = []
        self.odom_100hz_buffer = []
        self.bias_acc_buffer = []
        self.bias_gyro_buffer = []

    def imu_callback(self, imu_msg: Imu):
        if len(self.odom_100hz_buffer) == 0:
            return

        timestamp = self._stamp_to_sec(imu_msg.header.stamp)
        self.imu_buffer.append((timestamp, imu_msg))
        if len(self.imu_buffer) > 2000:
            self.imu_buffer.pop(0)

        if self.imu_buffer[-1][0] <= self.odom_100hz_buffer[-1][0] + 0.010:
            return

        start_idx = None
        for i in range(len(self.imu_buffer)):
            if self.imu_buffer[-(i + 1)][0] > self.odom_100hz_buffer[-1][0]:
                start_idx = -(i + 1)
            else:
                break

        if start_idx is None:
            return

        for i in range(start_idx, 0):
            imu_ts = self.imu_buffer[i][0]
            bias_acc = self._lookup_bias(self.bias_acc_buffer, imu_ts)
            bias_gyro = self._lookup_bias(self.bias_gyro_buffer, imu_ts)
            self.odom_100hz_buffer.append(
                integrate(
                    self.odom_100hz_buffer[-1],
                    self.imu_buffer[i],
                    bias_acc=bias_acc,
                    bias_gyro=bias_gyro,
                )
            )
            if len(self.odom_100hz_buffer) > 1000:
                self.odom_100hz_buffer.pop(0)

        self.odom_pub.publish(self.odom_100hz_buffer[-1][1])

    def odom_callback(self, msg: Odometry):
        timestamp = self._stamp_to_sec(msg.header.stamp)
        self.odom_10hz_buffer.append((timestamp, msg))
        if len(self.odom_10hz_buffer) > 100:
            self.odom_10hz_buffer.pop(0)

        while self.odom_100hz_buffer and self.odom_100hz_buffer[-1][0] > timestamp:
            self.odom_100hz_buffer.pop()
        self.odom_100hz_buffer.append((timestamp, msg))

    def bias_acc_callback(self, msg: Vector3Stamped):
        timestamp = self._stamp_to_sec(msg.header.stamp)
        bias_acc = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        self.bias_acc_buffer.append((timestamp, bias_acc))
        if len(self.bias_acc_buffer) > 2000:
            self.bias_acc_buffer.pop(0)

    def bias_gyro_callback(self, msg: Vector3Stamped):
        timestamp = self._stamp_to_sec(msg.header.stamp)
        bias_gyro = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        self.bias_gyro_buffer.append((timestamp, bias_gyro))
        if len(self.bias_gyro_buffer) > 2000:
            self.bias_gyro_buffer.pop(0)

    @staticmethod
    def _lookup_bias(buffer, timestamp):
        if len(buffer) == 0:
            return None
        best_idx = None
        best_dt = float("inf")
        for i, (ts, _) in enumerate(buffer):
            dt = abs(ts - timestamp)
            if dt < best_dt:
                best_dt = dt
                best_idx = i
            elif ts > timestamp and dt > best_dt:
                # Buffer is timestamp-ordered; once error grows on future side, it won't get better.
                break
        if best_idx is None:
            return None
        return buffer[best_idx][1]

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ImuPropagatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

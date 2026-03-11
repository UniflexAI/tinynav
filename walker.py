#!/usr/bin/env python3
# walker.py - 机器人行走控制器
import argparse
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import numpy as np
import cv2

class RobotWalker(Node):
    def __init__(self):
        super().__init__('robot_walker')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actionPublisher = self.create_publisher(String, '/service/command', 10)
        self._robot_status = ''
        self._pose = None
        self._latest_image_msg = None
        self.create_subscription(String, '/robot_status', self._robot_status_cb, 10)
        self.create_subscription(PoseStamped, '/insight/vio_pose', self._pose_cb, 10)
        self.create_subscription(Image, '/insight/camera_left_rectified', self._image_cb, 10)
        self.get_logger().info('Robot walker initialized')

    def _robot_status_cb(self, msg):
        self._robot_status = msg.data.strip()

    def _pose_cb(self, msg):
        self._pose = msg.pose

    def _image_cb(self, msg):
        self._latest_image_msg = msg

    def save_camera_snapshot(self, path='current.jpg'):
        """将最近一帧 /insight/camera_left_rectified 保存为 path，存在则覆盖。返回是否成功。"""
        if self._latest_image_msg is None:
            self.get_logger().warn('No image received yet')
            return False
        msg = self._latest_image_msg
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            step = msg.step if msg.step else msg.width * 3
            if msg.encoding in ('rgb8', 'bgr8'):
                img = data.reshape((msg.height, step))[:, : msg.width * 3].reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'mono8':
                img = data.reshape((msg.height, step))[:, : msg.width]
            else:
                ch = 3 if 'rgb' in msg.encoding or 'bgr' in msg.encoding else 1
                img = data.reshape((msg.height, step))[:, : msg.width * ch]
                if img.ndim == 2 and ch == 3:
                    img = img.reshape((msg.height, msg.width, 3))
            cv2.imwrite(path, img)
            self.get_logger().info(f'Saved snapshot to {path}')
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to save snapshot: {e}')
            return False

    def _yaw_from_pose(self, pose):
        """从 geometry_msgs/Pose 的 orientation 四元数得到 yaw (弧度)"""
        if pose is None:
            return None
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _angle_diff(self, a, b):
        """两角度之差 (弧度)，范围 [-pi, pi]"""
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    def playStandUp(self, timeout_sec=30.0, command_rate=2.0):
        """发送站立命令，直到 /robot_status 为 standup 或超时"""
        self.get_logger().info('Playing stand up, waiting for standup status...')
        period = 1.0 / command_rate
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            msg = String()
            msg.data = 'play stand'
            self.actionPublisher.publish(msg)
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=period / 5)
            if self._robot_status == 'standup':
                self.get_logger().info('Stand up completed')
                return
        self.get_logger().warn('Stand up timeout')

    def playSitDown(self, timeout_sec=30.0, command_rate=2.0):
        """发送坐下命令，直到 /robot_status 为 sitting 或超时"""
        self.get_logger().info('Playing sit down, waiting for sitting status...')
        period = 1.0 / command_rate
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            msg = String()
            msg.data = 'play sit'
            self.actionPublisher.publish(msg)
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=period / 5)
            if self._robot_status == 'sitting':
                self.get_logger().info('Sit down completed')
                return
        self.get_logger().warn('Sit down timeout')
        
    def move_linear(self, x, y, duration, rate=50, pose_timeout=2.0):
        """直线移动：优先根据 self._pose 位移达到目标距离后停止，无 pose 或超时则按 duration 停止"""
        msg = Twist()
        msg.linear.x = float(x)
        msg.linear.y = float(y)
        target_distance = duration * math.sqrt(x * x + y * y)
        start_time = time.time()
        deadline = start_time + duration + pose_timeout
        x0, y0 = None, None
        if self._pose is not None:
            x0 = self._pose.position.x
            y0 = self._pose.position.y

        while rclpy.ok() and time.time() < deadline:
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=1.0 / rate)
            if self._pose is not None and x0 is not None and y0 is not None:
                dx = self._pose.position.x - x0
                dy = self._pose.position.y - y0
                traveled = math.sqrt(dx * dx + dy * dy)
                if traveled >= target_distance:
                    break
            if x0 is None and (time.time() - start_time) >= duration:
                break

        msg.linear.x = 0.0
        msg.linear.y = 0.0
        self.publisher.publish(msg)

    def turn(self, angular_z, duration, rate=50, pose_timeout=2.0):
        """旋转：优先根据 self._pose 的 yaw 变化达到目标角度后停止，无 pose 或超时则按 duration 停止"""
        msg = Twist()
        msg.angular.z = float(angular_z)
        target_angle = abs(angular_z * duration)
        start_time = time.time()
        deadline = start_time + duration + pose_timeout
        start_yaw = self._yaw_from_pose(self._pose)

        while rclpy.ok() and time.time() < deadline:
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=1.0 / rate)
            current_yaw = self._yaw_from_pose(self._pose)
            if start_yaw is not None and current_yaw is not None:
                delta = abs(self._angle_diff(current_yaw, start_yaw))
                if delta >= target_angle:
                    break
            if start_yaw is None and (time.time() - start_time) >= duration:
                break

        msg.angular.z = 0.0
        self.publisher.publish(msg)
        self.get_logger().info('Turned')

def main():
    parser = argparse.ArgumentParser(description='Robot Walker Controller')
    parser.add_argument('--direction', choices=['forward', 'back', 'left', 'right'], default='forward')
    parser.add_argument('--distance', type=float, default=0, help='Distance in meters')
    parser.add_argument('--speed', type=float, default=0.3, help='Speed in m/s')
    parser.add_argument('--turn', type=float, default=0, help='Turn angle in degrees (positive=left, negative=right)')
    parser.add_argument('--turn-speed', type=float, default=0.5, help='Turn speed rad/s')
    parser.add_argument('--stand', action='store_true', help='Stand up')
    parser.add_argument('--sit', action='store_true', help='Sit down')
    parser.add_argument('--snapshot', action='store_true', help='Save one frame from /insight/camera_left_raw to current.jpg')
    
    args = parser.parse_args()
    
    rclpy.init()
    walker = RobotWalker()
    
    # 若有 --snapshot：等一帧图像并保存为当前目录 current.jpg（覆盖）
    if args.snapshot:
        deadline = time.time() + 10.0
        while rclpy.ok() and walker._latest_image_msg is None and time.time() < deadline:
            rclpy.spin_once(walker, timeout_sec=0.1)
        if walker.save_camera_snapshot('current.jpg'):
            pass
        else:
            walker.get_logger().warn('Snapshot skipped (no image within 10s)')
    
    # 站/坐 独立控制
    if args.stand:
        walker.playStandUp()
    if args.sit:
        walker.playSitDown()
    
    # 处理移动（只有明确指定了移动方向时才移动）
    move_only = False
    if args.direction == 'forward' and args.distance > 0:
        duration = args.distance / args.speed
        walker.move_linear(args.speed, 0, duration)
        move_only = True
    elif args.direction == 'back' and args.distance > 0:
        duration = args.distance / args.speed
        walker.move_linear(-args.speed, 0, duration)
        move_only = True
    elif args.direction == 'left' and args.distance > 0:
        duration = args.distance / args.speed
        walker.move_linear(0, args.speed, duration)
        move_only = True
    elif args.direction == 'right' and args.distance > 0:
        duration = args.distance / args.speed
        walker.move_linear(0, -args.speed, duration)
        move_only = True
    
    # 处理旋转
    if args.turn != 0:
        angle_rad = abs(args.turn) * 3.14159 / 180
        direction = 1 if args.turn > 0 else -1
        duration = angle_rad / args.turn_speed
        walker.turn(direction * args.turn_speed, duration)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()

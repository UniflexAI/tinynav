#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Header

class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_publisher')
        
        # Camera parameters
        self.width = 848
        self.height = 480
        self.fx = 423.99843311309814  # focal length x
        self.fy = 423.99847984313965  # focal length y
        self.cx = 424.0  # principal point x
        self.cy = 240.0  # principal point y
        
        # Stereo baseline (distance between cameras)
        self.baseline = 0.051  # 51mm = 0.051m (from SDF positions)
        
        # Create publishers
        self.left_camera_info_pub = self.create_publisher(
            CameraInfo, '/camera/camera/infra1/camera_info', 10)
        self.right_camera_info_pub = self.create_publisher(
            CameraInfo, '/camera/camera/infra2/camera_info', 10)
        
        # Create timer to publish camera_info
        self.timer = self.create_timer(0.1, self.publish_camera_info)  # 10Hz
        
        self.get_logger().info(f'CameraInfo publisher started with baseline: {self.baseline}m')
    
    def create_camera_info(self, is_right_camera=False):
        """Create camera_info message with proper stereo calibration"""
        msg = CameraInfo()
        msg.header = Header()
        msg.header.frame_id = f'vehicle_blue/infra{"2" if is_right_camera else "1"}_link/infra{"2" if is_right_camera else "1"}'
        
        msg.height = self.height
        msg.width = self.width
        
        # Distortion model
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        
        # Intrinsic matrix K
        msg.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0
        ]
        
        # Rotation matrix R (identity for rectified images)
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        
        # Projection matrix P
        if is_right_camera:
            # For right camera, encode baseline in P[3] (Tx component)
            # P[3] = -fx * baseline (negative because right camera is to the right)
            tx = -self.fx * self.baseline
            msg.p = [
                self.fx, 0.0, self.cx, tx,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        else:
            # For left camera, no baseline offset
            msg.p = [
                self.fx, 0.0, self.cx, 0.0,
                0.0, self.fy, self.cy, 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        
        # Binning and ROI
        msg.binning_x = 0
        msg.binning_y = 0
        msg.roi.x_offset = 0
        msg.roi.y_offset = 0
        msg.roi.height = 0
        msg.roi.width = 0
        msg.roi.do_rectify = False
        
        return msg
    
    def publish_camera_info(self):
        """Publish camera_info for both cameras"""
        # Get current time
        now = self.get_clock().now()
        
        # Create and publish left camera info
        left_msg = self.create_camera_info(is_right_camera=False)
        left_msg.header.stamp = now.to_msg()
        self.left_camera_info_pub.publish(left_msg)
        
        # Create and publish right camera info
        right_msg = self.create_camera_info(is_right_camera=True)
        right_msg.header.stamp = now.to_msg()
        self.right_camera_info_pub.publish(right_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 

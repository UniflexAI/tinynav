from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
import numpy as np
import cv2
from codetiming import Timer

class LooperHelpNode(Node):

    def __init__(self):
        super().__init__("LooperHelpNode")
        self.conbined_image = self.create_subscription(Image, "/image_combine_raw", self.conbined_image_callback, 10)
        self.left_image_pub = self.create_publisher(Image, "/camera/camera/infra1/image_rect_raw", 10)
        self.right_image_pub = self.create_publisher(Image, "/camera/camera/infra2/image_rect_raw", 10)
        self.left_camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera/infra1/camera_info", 10)
        self.right_camera_info_pub = self.create_publisher(CameraInfo, "/camera/camera/infra2/camera_info", 10)

        # Subscribe to IMU data with BEST_EFFORT QoS for high-frequency sensor data
        imu_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=3000)
        self.imu_sub = self.create_subscription(Imu, "/imu_data_looper", self.imu_callback, imu_qos_profile)
        self.get_logger().info(f"IMU transform setup completed. Subscribed to /imu_data_looper, publishing to /camera/camera/imu")
        self.imu_pub = self.create_publisher(Imu, "/camera/camera/imu", 10)
        self.bridge = CvBridge()
        # Parse camera intrinsics from the configuration
        self._setup_camera_parameters()
        self._setup_rectification_maps()
        self._setup_imu_transform()

    def _setup_camera_parameters(self):
        """Parse and setup camera intrinsics and distortion parameters."""
        # cam0 (left camera)
        self.left_intrinsics = np.array([
            [328.5085990437148, 0.0, 270.6218092863068],
            [0.0, 328.52132652118473, 319.51821945478923],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.left_distortion = np.array([
            -0.021988105453288265,
            0.008703225038080223,
            -0.010822473128295644,
            0.001967351487176878
        ], dtype=np.float64)
        self.left_resolution = (544, 640)  # width, height (width=544, height=640)
        # cam1 (right camera)
        self.right_intrinsics = np.array([
            [331.38399517340775, 0.0, 270.5789045245627],
            [0.0, 331.3716258633944, 320.3343307678317],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.right_distortion = np.array([
            -0.019734927308809148, 
            0.002986313775804545, 
            -0.0035613512769424303, 
            -0.0006058839364257485
        ], dtype=np.float64)
        
        self.right_resolution = (544, 640)  # width, height (width=544, height=640)
        
        # Stereo transformation (T_cn_cnm1: transformation from cam0 to cam1)
        self.T_cam0_to_cam1 = np.array([
            [0.9999961336043292, -0.0021508512002941923, -0.0017625593636585392, -0.09984359777501882],
            [0.0021507748583585694, 0.999997686059224, -4.520740780265428e-05, -0.0005906763650592106],
            [0.001762652519607889, 4.141636464720712e-05, 0.9999984456691837, -2.220462710896233e-05],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # T_cam_imu for cam0 (transformation from camera to IMU)
        self.T_cam0_imu = np.array([
            [0.0023948604158394726, -0.9999948996006053, -0.0021131531805036402, 0.040954475389185244],
            [0.999984594298381, 0.002405406876127003, -0.005002517732401834, 0.011770168453266817],
            [0.005007575210754112, -0.0021011402941994023, 0.9999852545912731, -0.01859562066910737],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.get_logger().info("Camera parameters loaded successfully")
    
    def _setup_rectification_maps(self):
        """Compute rectification maps for stereo cameras."""
        # Extract rotation and translation from transformation matrix
        # T_cam0_to_cam1: transformation from cam0 (left) to cam1 (right)
        # For stereoRectify, we need R_rl (rotation from right to left) and t_rl (translation from right to left)
        R_lr = self.T_cam0_to_cam1[:3, :3]  # Rotation from left to right
        t_lr = self.T_cam0_to_cam1[:3, 3]   # Translation from left to right
        
        # Convert to right-to-left (R_rl = R_lr^T, t_rl = -R_lr^T * t_lr)
        R_rl = R_lr.T
        t_rl = -R_rl.T @ t_lr
        
        # Stereo rectification for fisheye cameras
        flags = cv2.fisheye.CALIB_ZERO_DISPARITY
        
        # Input image size (stereo_input_width, stereo_input_height)
        stereo_input_size = self.left_resolution  # (width, height) -> (544, 640)
        stereo_input_size_wh = (stereo_input_size[0], stereo_input_size[1])  # OpenCV expects (width, height)
        
        # Output/rectified image size (model_input_w, model_input_h)
        # Can be same as input or different - using same size here
        model_input_size_wh = stereo_input_size_wh
        
        # Prepare camera matrices and distortion coefficients for fisheye stereo rectify
        Kl = self.left_intrinsics   # Left camera (cam0)
        Dl = self.left_distortion
        Kr = self.right_intrinsics   # Right camera (cam1)
        Dr = self.right_distortion
        
        # Compute stereo rectification - matching C++ signature
        # cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr, input_size, R_rl, t_rl, Rl, Rr, Pl, Pr, Q, flags, new_size, balance, fov_scale)
        Rl, Rr, Pl, Pr, Q = cv2.fisheye.stereoRectify(
            Kl, Dl, Kr, Dr,
            stereo_input_size_wh,
            R_rl, t_rl,
            flags,
            model_input_size_wh,
            0.0,  # balance
            1.0   # fov_scale
        )
        
        # Store rectified camera matrices (Pl and Pr are 3x4 projection matrices)
        # Extract 3x3 intrinsics from the first 3 columns
        self.left_rectified_intrinsics = Pl[:, :3]  # 3x3 intrinsics matrix
        self.left_rectified_projection = Pl  # 3x4 projection matrix
        self.right_rectified_intrinsics = Pr[:, :3]  # 3x3 intrinsics matrix
        self.right_rectified_projection = Pr  # 3x4 projection matrix
        self.right_rectified_projection[0, 3] = -1.0 * self.right_rectified_projection[0, 3]
        
        # Store rectified image size
        self.rectified_width = model_input_size_wh[0]
        self.rectified_height = model_input_size_wh[1]
        
        # Compute rectification maps using the rectified camera matrices
        self.left_map1, self.left_map2 = cv2.fisheye.initUndistortRectifyMap(
            Kl, Dl, Rl, Pl, model_input_size_wh, cv2.CV_16SC2
        )
        
        self.right_map1, self.right_map2 = cv2.fisheye.initUndistortRectifyMap(
            Kr, Dr, Rr, Pr, model_input_size_wh, cv2.CV_16SC2
        )
        
        self.get_logger().info("Rectification maps computed successfully")
    
    def _setup_imu_transform(self):
        """Setup IMU to camera transform."""
        # T_cam0_imu: transformation from cam0 to IMU
        # Compute T_imu_cam0: transformation from IMU to cam0 (inverse)
        T_cam0_imu = np.linalg.inv(self.T_cam0_imu)
        R_cam0_imu = T_cam0_imu[:3, :3]
        t_cam0_imu = T_cam0_imu[:3, 3]
        
        # T_imu_cam0 = inv(T_cam0_imu)
        # R_imu_cam0 = R_cam0_imu^T
        # t_imu_cam0 = -R_cam0_imu^T @ t_cam0_imu
        self.R_imu_cam0 = R_cam0_imu.T
        self.t_imu_cam0 = -self.R_imu_cam0 @ t_cam0_imu
        
    
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions (w, x, y, z)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
    
    def imu_callback(self, imu_msg: Imu):
        """Transform IMU data from IMU frame to cam0 frame."""
        # Extract IMU data
        acc_imu = np.array([imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z])
        
        ang_vel_imu = np.array([imu_msg.angular_velocity.x,
                                imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z])
        
        # Transform linear acceleration: R_imu_cam0 @ acc_imu
        acc_cam0 = self.R_imu_cam0 @ acc_imu
        
        # Transform angular velocity: R_imu_cam0 @ ang_vel_imu
        ang_vel_cam0 = self.R_imu_cam0 @ ang_vel_imu
        
        # Transform orientation quaternion
        # Extract IMU orientation quaternion (ROS uses x, y, z, w)
        q_imu_ros = np.array([imu_msg.orientation.x,
                              imu_msg.orientation.y,
                              imu_msg.orientation.z,
                              imu_msg.orientation.w])
        
        # Convert to (w, x, y, z) format
        q_imu = np.array([q_imu_ros[3], q_imu_ros[0], q_imu_ros[1], q_imu_ros[2]])
        
        # Convert rotation matrix to quaternion
        q_imu_cam0 = self._rotation_matrix_to_quaternion(self.R_imu_cam0)
        
        # Compose quaternions: q_cam0 = q_imu_cam0 * q_imu
        q_cam0 = self._quaternion_multiply(q_imu_cam0, q_imu)
        
        # Convert back to ROS format (x, y, z, w)
        q_cam0_ros = np.array([q_cam0[1], q_cam0[2], q_cam0[3], q_cam0[0]])
        
        # Create transformed IMU message
        imu_cam0_msg = Imu()
        imu_cam0_msg.header = imu_msg.header
        imu_cam0_msg.header.frame_id = "camera_imu_frame"  # Update frame_id to cam0 frame
        
        # Set transformed linear acceleration
        imu_cam0_msg.linear_acceleration.x = acc_cam0[0]
        imu_cam0_msg.linear_acceleration.y = acc_cam0[1]
        imu_cam0_msg.linear_acceleration.z = acc_cam0[2]
        #print(f"acc_cam0 : {acc_cam0}")
        
        # Set transformed angular velocity
        imu_cam0_msg.angular_velocity.x = ang_vel_cam0[0]
        imu_cam0_msg.angular_velocity.y = ang_vel_cam0[1]
        imu_cam0_msg.angular_velocity.z = ang_vel_cam0[2]
        
        # Set transformed orientation
        imu_cam0_msg.orientation.x = q_cam0_ros[0]
        imu_cam0_msg.orientation.y = q_cam0_ros[1]
        imu_cam0_msg.orientation.z = q_cam0_ros[2]
        imu_cam0_msg.orientation.w = q_cam0_ros[3]
        
        # Copy covariance matrices (assuming they remain valid in cam0 frame)
        imu_cam0_msg.linear_acceleration_covariance = imu_msg.linear_acceleration_covariance
        imu_cam0_msg.angular_velocity_covariance = imu_msg.angular_velocity_covariance
        imu_cam0_msg.orientation_covariance = imu_msg.orientation_covariance
        
        # Publish transformed IMU data
        self.imu_pub.publish(imu_cam0_msg)
    
    def undistort_and_rectify_left(self, image):
        """Undistort and rectify left camera image."""
        return cv2.remap(image, self.left_map1, self.left_map2, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT)
    
    def undistort_and_rectify_right(self, image):
        """Undistort and rectify right camera image."""
        return cv2.remap(image, self.right_map1, self.right_map2, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_CONSTANT)
    
    def _create_camera_info(self, header, intrinsics_3x3, projection_3x4, width, height):
        """Create a CameraInfo message from rectified camera parameters."""
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width = width
        camera_info.height = height
        
        # Set distortion model to "plumb_bob" (standard pinhole) after rectification
        # Rectified images have no distortion
        camera_info.distortion_model = "plumb_bob"
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion after rectification
        
        # Set intrinsics matrix K (3x3) - flattened row-major
        camera_info.k = intrinsics_3x3.flatten().tolist()
        
        # Set rectification matrix R (3x3 identity after rectification)
        camera_info.r = np.eye(3, dtype=np.float64).flatten().tolist()
        
        # Set projection matrix P (3x4) - flattened row-major
        camera_info.p = projection_3x4.flatten().tolist()
        
        return camera_info

    @Timer(name="Image processing", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def conbined_image_callback(self, conbined_image : Image):
        img_np = np.frombuffer(conbined_image.data, dtype=np.uint8)
        if conbined_image.encoding == "nv12":
            yuv_image = img_np.reshape(conbined_image.height // 2 * 3, conbined_image.width)
            gray_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2GRAY_NV12)
            half_height = conbined_image.height // 2
            left_image_raw = gray_image[:half_height, :]
            right_image_raw = gray_image[half_height:, :]
            
            # Undistort and rectify images
            left_image_rect = self.undistort_and_rectify_left(left_image_raw)
            right_image_rect = self.undistort_and_rectify_right(right_image_raw)
            
            # Publish rectified images
            left_msg = self.bridge.cv2_to_imgmsg(left_image_rect, encoding="mono8")
            left_msg.header = conbined_image.header
            self.left_image_pub.publish(left_msg)
            
            right_msg = self.bridge.cv2_to_imgmsg(right_image_rect, encoding="mono8")
            right_msg.header = conbined_image.header
            self.right_image_pub.publish(right_msg)
            
            # Publish camera info for rectified images
            left_camera_info = self._create_camera_info(
                conbined_image.header,
                self.left_rectified_intrinsics,
                self.left_rectified_projection,
                self.rectified_width,
                self.rectified_height
            )
            self.left_camera_info_pub.publish(left_camera_info)
            
            right_camera_info = self._create_camera_info(
                conbined_image.header,
                self.right_rectified_intrinsics,
                self.right_rectified_projection,
                self.rectified_width,
                self.rectified_height
            )
            #print(f"right_rectified_projection : {self.right_rectified_projection}")
            self.right_camera_info_pub.publish(right_camera_info)

            # Save images for debugging
            #cv2.imwrite("/tinynav/output/left_image_raw.png", left_image_raw)
            #cv2.imwrite("/tinynav/output/right_image_raw.png", right_image_raw)
            #cv2.imwrite("/tinynav/output/left_image_rect.png", left_image_rect)
            #cv2.imwrite("/tinynav/output/right_image_rect.png", right_image_rect)
            # concate the left and right image alone width axis
            combined_image = np.concatenate((left_image_rect, right_image_rect), axis=1)
            combined_image_color = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)
            # draw a line in the middle of the combined image hori
            line_thickness = 2
            line_color = (0, 0, 255)
            line_start_point = (0, combined_image.shape[0] // 2)
            line_end_point = (combined_image.shape[1], combined_image.shape[0] // 2)
            #cv2.line(combined_image_color, line_start_point, line_end_point, line_color, line_thickness)
            #cv2.imwrite("/tinynav/output/combined_image.png", combined_image_color)
'''
    cam0:
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: [-0.021988105453288265, 0.008703225038080223, -0.010822473128295644, 0.001967351487176878]
  distortion_model: equidistant
  intrinsics: [328.5085990437148, 328.52132652118473, 270.6218092863068, 319.51821945478923]
  resolution: [544, 640]
cam1:
  T_cn_cnm1:
  - [0.9999961336043292, -0.0021508512002941923, -0.0017625593636585392, -0.09984359777501882]
  - [0.0021507748583585694, 0.999997686059224, -4.520740780265428e-05, -0.0005906763650592106]
  - [0.001762652519607889, 4.141636464720712e-05, 0.9999984456691837, -2.220462710896233e-05]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: [0]
  camera_model: pinhole
  distortion_coeffs: [-0.019734927308809148, 0.002986313775804545, -0.0035613512769424303, -0.0006058839364257485]
  distortion_model: equidistant
  intrinsics: [331.38399517340775, 331.3716258633944, 270.5789045245627, 320.3343307678317]
  resolution: [544, 640]
  '''
'''
cam0:
  T_cam_imu:
  - [0.0023948604158394726, -0.9999948996006053, -0.0021131531805036402, 0.040954475389185244]
  - [0.999984594298381, 0.002405406876127003, -0.005002517732401834, 0.011770168453266817]
  - [0.005007575210754112, -0.0021011402941994023, 0.9999852545912731, -0.01859562066910737]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: [1]
  camera_model: pinhole
  distortion_coeffs: [-0.021988105453288265, 0.008703225038080223, -0.010822473128295644, 0.001967351487176878]
  distortion_model: equidistant
  intrinsics: [328.5085990437148, 328.52132652118473, 270.6218092863068, 319.51821945478923]
  resolution: [544, 640]
  rostopic: /image_left_raw
  timeshift_cam_imu: 0.0014778568599085925
cam1:
  T_cam_imu:
  - [0.00023520694286233157, -0.9999925035124195, -0.0038649187129484127, -0.05888182072765199]
  - [0.9999872048193281, 0.0002547324087163738, -0.0050522578148054975, 0.01126838936854612]
  - [0.005053204460670778, -0.003863680934500026, 0.9999797683424977, -0.018545120405647462]
  - [0.0, 0.0, 0.0, 1.0]
  T_cn_cnm1:
  - [0.9999961336043285, -0.0021508512002941923, -0.001762559363658539, -0.09984359777501882]
  - [0.0021507748583585694, 0.9999976860592233, -4.520740780265428e-05, -0.0005906763650592106]
  - [0.0017626525196078888, 4.141636464720712e-05, 0.9999984456691831, -2.220462710896233e-05]
  - [0.0, 0.0, 0.0, 1.0]
  cam_overlaps: [0]
  camera_model: pinhole
  distortion_coeffs: [-0.019734927308809148, 0.002986313775804545, -0.0035613512769424303, -0.0006058839364257485]
  distortion_model: equidistant
  intrinsics: [331.38399517340775, 331.3716258633944, 270.5789045245627, 320.3343307678317]
  resolution: [544, 640]
  rostopic: /image_right_raw
  timeshift_cam_imu: 0.0014553843531526387
'''


if __name__ == "__main__":
    rclpy.init()

    node = LooperHelpNode()
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    executor.shutdown()



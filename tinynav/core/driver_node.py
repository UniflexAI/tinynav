import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
from huggingface_hub import snapshot_download

class KittiDriverNode(Node):
    def __init__(self, seq_num:int = 5):
        super().__init__('kitti_driver')
        self.bridge = CvBridge()
        self.index = 0
        assert seq_num >= 0 and seq_num <= 21, "KITTI dataset consists 21 sequences only."
        pattern = f"sequences/{seq_num:02d}/**"
        dataset_root = snapshot_download(repo_id="Junlinp/test_dataset",repo_type="dataset", allow_patterns=pattern)

        # === Constants: Set your KITTI paths here ===
        self.dataset_path = f"{dataset_root}/sequences/{seq_num:02d}"
        self.calib_path = f"{dataset_root}/sequences/{seq_num:02d}/calib.txt"

        # Publishers
        self.left_pub = self.create_publisher(Image, '/camera/camera/infra1/image_rect_raw', 10)
        self.right_pub = self.create_publisher(Image, '/camera/camera/infra2/image_rect_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera/infra1/camera_info', 10)

        # Read calibration and prepare CameraInfo
        self.K, self.baseline = self.read_calib(self.calib_path)
        self.camera_info = self.create_camera_info_msg(self.K, self.baseline)

        # Timer to publish images every 0.1 sec (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_images)

    def read_calib(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        P0 = np.array([float(x) for x in lines[0].split()[1:]]).reshape(3, 4)
        P1 = np.array([float(x) for x in lines[1].split()[1:]]).reshape(3, 4)
        fx = P0[0, 0]
        baseline = abs(P0[0, 3] - P1[0, 3]) / fx
        return P0[:3, :3], baseline

    def create_camera_info_msg(self, K, baseline):
        msg = CameraInfo()
        msg.k = K.flatten().tolist()
        msg.p[0] = baseline
        msg.width = 1241
        msg.height = 376
        return msg

    def publish_images(self):
        left_path = os.path.join(self.dataset_path, "image_0", f"{self.index:06d}.png")
        right_path = os.path.join(self.dataset_path, "image_1", f"{self.index:06d}.png")
        if not os.path.exists(left_path):
            self.get_logger().info("End of dataset.")
            return

        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

        if left_img is None or right_img is None:
            self.get_logger().warning(f"Missing image at index {self.index}")
            return

        timestamp = self.get_clock().now().to_msg()

        left_msg = self.bridge.cv2_to_imgmsg(left_img, encoding="mono8")
        right_msg = self.bridge.cv2_to_imgmsg(right_img, encoding="mono8")

        left_msg.header.stamp = timestamp
        right_msg.header.stamp = timestamp
        self.camera_info.header.stamp = timestamp

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.info_pub.publish(self.camera_info)

        self.index += 1

def main(args=None):
    rclpy.init(args=args)
    node = KittiDriverNode(seq_num = 5)

    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

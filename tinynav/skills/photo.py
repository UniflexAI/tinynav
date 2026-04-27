#!/usr/bin/env python3
"""
photo — Capture a keyframe image and save it to disk.

Usage:
  ros2 run tinynav photo [--save-path /data/photos] [--timeout 5]

Subscribes to /slam/keyframe_image, saves the first received frame as a JPEG,
then exits. Blocks until an image is received or timeout expires.

Example:
  ros2 run tinynav photo
  ros2 run tinynav photo --save-path /tmp/captures --timeout 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class PhotoSkill(Node):
    def __init__(self, save_path: str, timeout: float):
        super().__init__("skill_photo")
        self.save_path = save_path
        self.timeout = timeout
        self.bridge = CvBridge()
        self.captured = False

        os.makedirs(save_path, exist_ok=True)
        self.create_subscription(Image, "/slam/keyframe_image", self._image_cb, 10)
        self.get_logger().info(f"photo: waiting for keyframe image (timeout={timeout}s) ...")

    def _image_cb(self, msg: Image) -> None:
        if self.captured:
            return  # Only save the first frame
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            # If mono, convert to BGR for imwrite
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            ts = int(time.time())
            fname = os.path.join(self.save_path, f"capture_{ts}.jpg")
            cv2.imwrite(fname, image)
            self.get_logger().info(f"photo: ✓ saved {fname}")
            self.captured = True
        except Exception as exc:
            self.get_logger().error(f"photo: failed to save image: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a keyframe photo (blocking)")
    parser.add_argument("--save-path", default="/tmp/tinynav_photos", help="Directory to save photos")
    parser.add_argument("--timeout", type=float, default=5.0, help="Max wait for image in seconds")
    args = parser.parse_args()

    rclpy.init()
    node = PhotoSkill(args.save_path, args.timeout)
    deadline = time.time() + args.timeout
    while rclpy.ok() and not node.captured and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node.captured:
        node.get_logger().warn("photo: timeout, no image received")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if node.captured else 2


if __name__ == "__main__":
    raise SystemExit(main())

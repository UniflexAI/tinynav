#!/usr/bin/env python3
import argparse
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from pidnet_trt import PIDNetTRT, floor_probability, make_overlay


class PIDNetSegmentationNode(Node):
    def __init__(self, args):
        super().__init__("pidnet_segmentation_node")
        self.bridge = CvBridge()
        self.runner = PIDNetTRT(args.engine)
        self.floor_channels = args.floor_channels
        self.threshold = float(args.threshold)
        self.alpha = float(args.overlay_alpha)
        self.ema_current_weight = float(args.ema_current_weight)
        self.hysteresis_on = float(args.hysteresis_on)
        self.hysteresis_off = float(args.hysteresis_off)
        self.morph_open_kernel = int(args.morph_open_kernel)
        self.morph_close_kernel = int(args.morph_close_kernel)
        self.ema_prob = None
        self.stable_floor_mask = None
        self.publish_period = 1.0 / args.publish_hz if args.publish_hz > 0.0 else 0.0
        self.last_publish_time = 0.0
        self.frame_count = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image,
            args.image_topic,
            self.image_callback,
            sensor_qos,
        )
        self.prob_pub = self.create_publisher(Image, args.prob_topic, 10)
        self.stable_prob_pub = self.create_publisher(Image, args.stable_prob_topic, 10)
        self.overlay_pub = self.create_publisher(Image, args.overlay_topic, 10)
        self.get_logger().info(
            f"PIDNet segmentation publishing {args.prob_topic}, {args.stable_prob_topic}, "
            f"and {args.overlay_topic} from {args.image_topic} at {args.publish_hz:.2f} Hz"
        )

    def _should_process(self):
        if self.publish_period <= 0.0:
            return True
        now = time.monotonic()
        if now - self.last_publish_time < self.publish_period:
            return False
        self.last_publish_time = now
        return True

    def _morph_mask(self, mask):
        mask_u8 = mask.astype(np.uint8) * 255
        if self.morph_open_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_open_kernel, self.morph_open_kernel),
            )
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
        if self.morph_close_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_close_kernel, self.morph_close_kernel),
            )
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        return mask_u8 >= 127

    def _stable_floor_probability(self, prob_full):
        prob_full = prob_full.astype(np.float32, copy=False)
        if self.ema_prob is None or self.ema_prob.shape != prob_full.shape:
            self.ema_prob = prob_full.copy()
        else:
            w = np.clip(self.ema_current_weight, 0.0, 1.0)
            self.ema_prob = (1.0 - w) * self.ema_prob + w * prob_full

        if self.stable_floor_mask is None or self.stable_floor_mask.shape != prob_full.shape:
            mask = self.ema_prob >= self.threshold
        else:
            mask = self.stable_floor_mask.copy()
            mask[self.ema_prob >= self.hysteresis_on] = True
            mask[self.ema_prob <= self.hysteresis_off] = False

        self.stable_floor_mask = self._morph_mask(mask)
        return self.stable_floor_mask.astype(np.float32)

    def image_callback(self, msg):
        if not self._should_process():
            return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            start = time.perf_counter()
            raw = self.runner.infer(image)
            prob = floor_probability(raw, self.floor_channels)
            overlay, prob_full = make_overlay(
                image,
                prob,
                alpha=self.alpha,
                threshold=self.threshold,
            )
            stable_prob_full = self._stable_floor_probability(prob_full)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            self.get_logger().error(f"PIDNet segmentation failed: {exc}")
            return

        prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
        stable_prob_u8 = np.clip(stable_prob_full * 255.0, 0, 255).astype(np.uint8)
        prob_msg = self.bridge.cv2_to_imgmsg(prob_u8, encoding="mono8")
        prob_msg.header = msg.header
        stable_prob_msg = self.bridge.cv2_to_imgmsg(stable_prob_u8, encoding="mono8")
        stable_prob_msg.header = msg.header
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header

        self.prob_pub.publish(prob_msg)
        self.stable_prob_pub.publish(stable_prob_msg)
        self.overlay_pub.publish(overlay_msg)
        self.frame_count += 1
        if self.frame_count == 1 or self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Published PIDNet segmentation frame={self.frame_count} "
                f"input={image.shape} output={raw.shape} elapsed_ms={elapsed_ms:.2f}"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Publish PIDNet floor segmentation as ROS Image topics.")
    parser.add_argument(
        "--engine",
        default="/tinynav/tinynav/models/pidnet_s_cityscapes_256x320_aarch64.plan",
        help="Target-device TensorRT engine path.",
    )
    parser.add_argument("--image-topic", default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--prob-topic", default="/segmentation/floor_prob")
    parser.add_argument("--stable-prob-topic", default="/segmentation/floor_prob_stable")
    parser.add_argument("--overlay-topic", default="/segmentation/floor_overlay")
    parser.add_argument("--publish-hz", type=float, default=5.0)
    parser.add_argument(
        "--floor-channels",
        default="0,1",
        help="Comma-separated floor class channels. Cityscapes: 0=road, 1=sidewalk.",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--ema-current-weight", type=float, default=0.3)
    parser.add_argument("--hysteresis-on", type=float, default=0.65)
    parser.add_argument("--hysteresis-off", type=float, default=0.35)
    parser.add_argument("--morph-open-kernel", type=int, default=3)
    parser.add_argument("--morph-close-kernel", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = PIDNetSegmentationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node


def _load_pois(pois_json_path: str) -> dict[str, Any]:
    if not os.path.exists(pois_json_path):
        raise FileNotFoundError(f"pois.json not found: {pois_json_path}")
    with open(pois_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("pois.json must be a dict of POI entries.")
    return data


def _sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    k, _ = item
    try:
        return (0, str(int(k)))
    except ValueError:
        return (1, k)


def _build_path_from_name(pois: dict[str, Any], poi_name: str, frame_id: str) -> Path:
    matches: list[tuple[str, Any]] = []
    for k, v in sorted(pois.items(), key=_sort_key):
        if not isinstance(v, dict):
            continue
        if v.get("name") != poi_name:
            continue
        pos = v.get("position")
        if not isinstance(pos, list) or len(pos) < 3:
            continue
        matches.append((k, v))

    if not matches:
        available_names = sorted(
            {
                str(v.get("name"))
                for v in pois.values()
                if isinstance(v, dict) and "name" in v
            }
        )
        raise ValueError(
            f"POI name '{poi_name}' not found in pois.json. "
            f"Available names: {available_names}"
        )

    msg = Path()
    msg.header.frame_id = frame_id
    for _, poi in matches:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pos = poi["position"]
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


class PoiPublisher(Node):
    def __init__(self, topic: str):
        super().__init__("poi_manager")
        self.publisher = self.create_publisher(Path, topic, 10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish POI(s) as nav_msgs/Path by name."
    )
    parser.add_argument(
        "--pois-json", required=True, help="Absolute path to pois.json."
    )
    parser.add_argument("--name", required=True, help="POI name to search for.")
    parser.add_argument(
        "--topic", default="/mapping/pois", help="ROS2 topic to publish."
    )
    parser.add_argument(
        "--frame-id", default="map", help="frame_id for Path and PoseStamped."
    )
    parser.add_argument(
        "--publish-seconds",
        type=float,
        default=1.0,
        help="How long to keep publishing (seconds) for discovery reliability.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Publish rate while active (Hz).",
    )
    args = parser.parse_args()

    pois = _load_pois(args.pois_json)
    path_msg = _build_path_from_name(pois, args.name, args.frame_id)

    rclpy.init()
    node = PoiPublisher(args.topic)
    try:
        rate_hz = max(args.rate, 0.5)
        timer_period = 1.0 / rate_hz
        end_time_ns = node.get_clock().now().nanoseconds + int(
            args.publish_seconds * 1e9
        )

        while rclpy.ok() and node.get_clock().now().nanoseconds < end_time_ns:
            now_msg = node.get_clock().now().to_msg()
            path_msg.header.stamp = now_msg
            for pose in path_msg.poses:
                pose.header.stamp = now_msg
            node.publisher.publish(path_msg)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(timer_period)

        node.get_logger().info(
            f"Published {len(path_msg.poses)} POI pose(s) to {args.topic} for name='{args.name}'."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
go — Navigate to a POI and block until arrival.

Usage:
  ros2 run tinynav go --poi <id_or_name> [--pois-path /path/to/pois.json]

Reads pois.json to resolve POI id/name → position, publishes the target to
/nav/target_poi, then blocks until /poi_arrived confirms arrival.

Example:
  ros2 run tinynav go --poi reception
  ros2 run tinynav go --poi 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def load_poi_position(poi_id: str, pois_path: Path) -> list:
    """Look up a POI position by id or name from pois.json."""
    if not pois_path.exists():
        raise FileNotFoundError(f"POI file not found: {pois_path}")
    data = json.loads(pois_path.read_text())
    for key, poi in data.items():
        if str(poi.get("id", key)) == poi_id or poi.get("name") == poi_id:
            return poi["position"]
    raise KeyError(f"POI '{poi_id}' not found in {pois_path}")


class GoSkill(Node):
    def __init__(self, poi_id: str, pois_path: str):
        super().__init__("skill_go")
        self.poi_id = poi_id
        self.arrived = False

        # Resolve POI position
        position = load_poi_position(poi_id, Path(pois_path))
        self.get_logger().info(f"POI '{poi_id}' → position {position}")

        # Publishers / subscribers
        self.target_pub = self.create_publisher(String, "/nav/target_poi", 10)
        self.create_subscription(String, "/poi_arrived", self._on_arrived, 10)

        # Wait for publisher to be ready, then send target
        self._publish_target(position)

    def _publish_target(self, position: list) -> None:
        """Publish the target POI to /nav/target_poi."""
        # Wait briefly for subscribers
        deadline = time.time() + 3.0
        while self.target_pub.get_subscription_count() == 0:
            if time.time() >= deadline:
                self.get_logger().warn("No subscriber on /nav/target_poi, publishing anyway")
                break
            time.sleep(0.1)

        msg = String()
        msg.data = json.dumps({"poi_id": self.poi_id, "position": position})
        self.target_pub.publish(msg)
        self.get_logger().info(f"Published target POI '{self.poi_id}', waiting for arrival...")

    def _on_arrived(self, msg: String) -> None:
        data = json.loads(msg.data)
        if str(data.get("poi_id")) == self.poi_id:
            self.get_logger().info(f"✓ Arrived at POI '{self.poi_id}'!")
            self.arrived = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Navigate to a POI (blocking)")
    parser.add_argument("--poi", required=True, help="POI id or name")
    parser.add_argument(
        "--pois-path",
        default="/tmp/tinynav_map/pois.json",
        help="Path to pois.json (default: /tmp/tinynav_map/pois.json)",
    )
    args = parser.parse_args()

    rclpy.init()
    try:
        node = GoSkill(args.poi, args.pois_path)
        while rclpy.ok() and not node.arrived:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 1
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

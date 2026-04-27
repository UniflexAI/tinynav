#!/usr/bin/env python3
"""
go — Navigate to a POI and block until arrival.

Usage:
  ros2 run tinynav go --poi <id_or_name> [--pois-path /path/to/pois.json]

Reads pois.json to resolve POI id/name → position, publishes a single POI
to /mapping/cmd_poi (same topic pub_pois uses), then blocks until /poi_arrived
confirms arrival.

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


def load_poi(poi_id: str, pois_path: Path) -> dict:
    """Look up a POI by id or name from pois.json, return the full POI dict."""
    if not pois_path.exists():
        raise FileNotFoundError(f"POI file not found: {pois_path}")
    data = json.loads(pois_path.read_text())
    for key, poi in data.items():
        if str(poi.get("id", key)) == poi_id or poi.get("name") == poi_id:
            return poi
    raise KeyError(f"POI '{poi_id}' not found in {pois_path}")


class GoSkill(Node):
    def __init__(self, poi: dict, poi_id: str):
        super().__init__("skill_go")
        self.poi_id = poi_id
        self.arrived = False

        # Reuse the same topic as pub_pois — sends a single-POI payload
        self.cmd_pub = self.create_publisher(String, "/mapping/cmd_pois", 10)
        self.create_subscription(String, "/poi_arrived", self._on_arrived, 10)

        # Wait for map_node subscriber, then publish
        deadline = time.time() + 3.0
        while self.cmd_pub.get_subscription_count() == 0:
            if time.time() >= deadline:
                self.get_logger().warn("No subscriber on /mapping/cmd_pois, publishing anyway")
                break
            time.sleep(0.1)

        # Send single POI as {"0": poi_dict} — map_node will set poi_index=0
        payload = json.dumps({"0": poi}, separators=(",", ":"))
        msg = String()
        msg.data = payload
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"Going to POI '{poi_id}', waiting for arrival...")

    def _on_arrived(self, msg: String) -> None:
        data = json.loads(msg.data)
        # Accept arrival for any POI (single-target mode)
        self.get_logger().info(f"✓ Arrived at POI '{data.get('poi_index')}'!")
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

    try:
        poi = load_poi(args.poi, Path(args.pois_path))
    except (FileNotFoundError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rclpy.init()
    node = GoSkill(poi, args.poi)
    while rclpy.ok() and not node.arrived:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class SequentialPoiPublisher(Node):
    def __init__(
        self,
        pois: list[dict],
        topic: str,
        pose_topic: str,
        republish_interval_sec: float,
        arrival_xy_threshold: float,
        arrival_z_threshold: float,
    ):
        super().__init__("pub_pois")
        self.pois = pois
        self.arrival_xy_threshold = arrival_xy_threshold
        self.arrival_z_threshold = arrival_z_threshold

        self.current_position = None
        self.current_index = 0

        self.publisher = self.create_publisher(String, topic, 10)
        self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)
        self.timer = self.create_timer(republish_interval_sec, self.timer_callback)

        self.get_logger().info(
            f"Loaded {len(self.pois)} POIs. Waiting for pose on {pose_topic}."
        )
        if self.pois:
            self.publish_poi()

    def pose_callback(self, msg: Odometry):
        self.current_position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        self.advance_if_arrived()

    def timer_callback(self):
        if self.current_index >= len(self.pois):
            self.get_logger().info("All POIs have been sent and reached.")
            self.timer.cancel()
            self.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            return

        if self.current_position is not None:
            self.advance_if_arrived()
        if self.current_index < len(self.pois):
            self.publish_poi()

    def advance_if_arrived(self):
        while self.current_position is not None and self.current_index < len(self.pois):
            target_position = np.array(
                self.pois[self.current_index]["position"],
                dtype=np.float32,
            )
            diff_xy = np.linalg.norm(target_position[:2] - self.current_position[:2])
            diff_z = abs(float(target_position[2] - self.current_position[2]))
            if diff_xy >= self.arrival_xy_threshold or diff_z >= self.arrival_z_threshold:
                break

            poi_name = self.pois[self.current_index]["name"]
            self.get_logger().info(
                f"Reached {poi_name}: xy={diff_xy:.3f}m z={diff_z:.3f}m. Moving to next POI."
            )
            self.current_index += 1

    def publish_poi(self):
        poi = self.pois[self.current_index]
        msg = String()
        msg.data = json.dumps(
            {
                "0": {
                    "id": int(poi["id"]),
                    "name": str(poi["name"]),
                    "position": [float(v) for v in poi["position"]],
                }
            }
        )
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published POI {self.current_index + 1}/{len(self.pois)}: {poi['name']}"
        )


def load_pois(tinynav_map_path: Path, pois_path: Path | None) -> list[dict]:
    resolved_pois_path = pois_path or tinynav_map_path / "pois.json"
    if not resolved_pois_path.exists():
        raise FileNotFoundError(f"POI file not found: {resolved_pois_path}")

    data = json.loads(resolved_pois_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"POI file must contain a JSON object: {resolved_pois_path}")

    pois = []
    for key in sorted(data.keys(), key=int):
        poi = data[key]
        if "position" not in poi:
            raise ValueError(f"POI entry {key} is missing 'position'")
        pois.append(
            {
                "id": poi.get("id", int(key)),
                "name": poi.get("name", f"POI_{key}"),
                "position": poi["position"],
            }
        )
    return pois


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish TinyNav POIs one by one and advance only after the robot reaches each POI in map coordinates."
    )
    parser.add_argument(
        "--tinynav_map_path",
        type=Path,
        required=True,
        help="TinyNav map directory containing pois.json",
    )
    parser.add_argument(
        "--pois-path",
        type=Path,
        default=None,
        help="Optional explicit pois.json path. Defaults to <map-path>/pois.json",
    )
    parser.add_argument(
        "--topic",
        default="/mapping/cmd_pois",
        help="ROS topic to publish POIs to",
    )
    parser.add_argument(
        "--pose-topic",
        default="/mapping/current_pose_in_map",
        help="Robot pose topic in map coordinates",
    )
    parser.add_argument(
        "--republish-interval-sec",
        type=float,
        default=1.0,
        help="How often to retry publishing the active POI until it is reached",
    )
    parser.add_argument(
        "--arrival-xy-threshold",
        type=float,
        default=0.5,
        help="XY distance threshold in meters for considering a POI reached",
    )
    parser.add_argument(
        "--arrival-z-threshold",
        type=float,
        default=2.0,
        help="Z distance threshold in meters for considering a POI reached",
    )
    return parser.parse_args()


def main(args=None):
    parsed_args = parse_args()
    pois = load_pois(parsed_args.tinynav_map_path, parsed_args.pois_path)

    rclpy.init(args=args)
    node = SequentialPoiPublisher(
        pois=pois,
        topic=parsed_args.topic,
        pose_topic=parsed_args.pose_topic,
        republish_interval_sec=parsed_args.republish_interval_sec,
        arrival_xy_threshold=parsed_args.arrival_xy_threshold,
        arrival_z_threshold=parsed_args.arrival_z_threshold,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

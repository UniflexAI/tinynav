import argparse
import csv
import json
import os
from collections import defaultdict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tinynav.core.latency_trace import TRACE_TOPIC, parse_trace_event


STAGE_EVENTS = [
    ("backend", "manual_target_requested"),
    ("backend", "target_pose_published"),
    ("planning", "target_pose_received"),
    ("planning", "planning_input_received"),
    ("planning", "trajectory_published"),
    ("cmd_vel_control", "path_received"),
    ("cmd_vel_control", "cmd_vel_published"),
    ("unitree_control", "cmd_vel_received"),
    ("unitree_control", "robot_command_sent"),
]

SUMMARY_SEGMENTS = [
    (
        "backend_publish_ms",
        ("backend", "manual_target_requested"),
        ("backend", "target_pose_published"),
    ),
    (
        "backend_to_planning_ms",
        ("backend", "target_pose_published"),
        ("planning", "target_pose_received"),
    ),
    (
        "planning_wait_input_ms",
        ("planning", "target_pose_received"),
        ("planning", "planning_input_received"),
    ),
    (
        "planning_compute_ms",
        ("planning", "planning_input_received"),
        ("planning", "trajectory_published"),
    ),
    (
        "path_delivery_ms",
        ("planning", "trajectory_published"),
        ("cmd_vel_control", "path_received"),
    ),
    (
        "cmd_generation_ms",
        ("cmd_vel_control", "path_received"),
        ("cmd_vel_control", "cmd_vel_published"),
    ),
    (
        "cmd_to_unitree_ms",
        ("cmd_vel_control", "cmd_vel_published"),
        ("unitree_control", "cmd_vel_received"),
    ),
    (
        "unitree_dispatch_ms",
        ("unitree_control", "cmd_vel_received"),
        ("unitree_control", "robot_command_sent"),
    ),
    (
        "total_ms",
        ("backend", "manual_target_requested"),
        ("unitree_control", "robot_command_sent"),
    ),
]


class LatencyTraceRecorder(Node):
    def __init__(self, csv_path: str | None, summary_csv_path: str | None):
        super().__init__("latency_trace_recorder")
        self.events_by_trace = defaultdict(dict)
        self.summary_written = set()
        self.event_writer = None
        self.summary_writer = None
        self.event_file = None
        self.summary_file = None

        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            self.event_file = open(csv_path, "w", newline="")
            self.event_writer = csv.DictWriter(
                self.event_file,
                fieldnames=[
                    "trace_id",
                    "stage",
                    "event",
                    "t_ros_ns",
                    "t_wall_ns",
                    "source_stamp_ns",
                    "payload_json",
                ],
            )
            self.event_writer.writeheader()

        if summary_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)), exist_ok=True)
            self.summary_file = open(summary_csv_path, "w", newline="")
            self.summary_writer = csv.DictWriter(
                self.summary_file,
                fieldnames=["trace_id"] + [name for name, _, _ in SUMMARY_SEGMENTS],
            )
            self.summary_writer.writeheader()

        self.create_subscription(String, TRACE_TOPIC, self.trace_callback, 100)
        self.get_logger().info(f"Recording latency traces from {TRACE_TOPIC}")

    def destroy_node(self):
        if self.event_file:
            self.event_file.close()
        if self.summary_file:
            self.summary_file.close()
        super().destroy_node()

    def trace_callback(self, msg: String):
        event = parse_trace_event(msg.data)
        if not event:
            return
        trace_id = event.get("trace_id")
        stage = event.get("stage")
        event_name = event.get("event")
        if not trace_id or not stage or not event_name:
            return

        key = (stage, event_name)
        self.events_by_trace[trace_id][key] = event
        if self.event_writer:
            self.event_writer.writerow({
                "trace_id": trace_id,
                "stage": stage,
                "event": event_name,
                "t_ros_ns": event.get("t_ros_ns"),
                "t_wall_ns": event.get("t_wall_ns"),
                "source_stamp_ns": event.get("source_stamp_ns"),
                "payload_json": json.dumps(event, sort_keys=True),
            })
            self.event_file.flush()

        summary = self._build_summary(trace_id)
        if summary:
            line = " ".join(
                f"{name}={value:.1f}"
                for name, value in summary.items()
                if name != "trace_id" and value is not None
            )
            self.get_logger().info(f"{trace_id} {line}")
            if (
                self.summary_writer
                and trace_id not in self.summary_written
                and self._has_full_summary(trace_id)
            ):
                self.summary_writer.writerow(summary)
                self.summary_file.flush()
                self.summary_written.add(trace_id)

    def _build_summary(self, trace_id: str) -> dict | None:
        events = self.events_by_trace[trace_id]
        if not all(key in events for key in STAGE_EVENTS[:7]):
            return None
        summary = {"trace_id": trace_id}
        for name, start_key, end_key in SUMMARY_SEGMENTS:
            start = events.get(start_key)
            end = events.get(end_key)
            if start is None or end is None:
                summary[name] = None
                continue
            summary[name] = (int(end["t_ros_ns"]) - int(start["t_ros_ns"])) / 1e6
        return summary

    def _has_full_summary(self, trace_id: str) -> bool:
        events = self.events_by_trace[trace_id]
        return all(key in events for key in STAGE_EVENTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/tmp/tinynav_latency_trace_events.csv")
    parser.add_argument("--summary-csv", default="/tmp/tinynav_latency_trace_summary.csv")
    args = parser.parse_args()

    rclpy.init()
    node = LatencyTraceRecorder(args.csv, args.summary_csv)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

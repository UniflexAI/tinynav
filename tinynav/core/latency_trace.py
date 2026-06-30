import json
import time
from typing import Any

from builtin_interfaces.msg import Time as RosTime
from rclpy.node import Node
from std_msgs.msg import String


TRACE_TOPIC = "/debug/latency_trace"
TRACE_PREFIX = "manual_target:"


def make_trace_id() -> str:
    return f"manual-{time.time_ns()}"


def make_trace_publisher(node: Node):
    return node.create_publisher(String, TRACE_TOPIC, 10)


def ros_time_ns(stamp: RosTime | None) -> int | None:
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def node_time_ns(node: Node) -> int:
    return ros_time_ns(node.get_clock().now().to_msg()) or time.time_ns()


def encode_trace_frame(trace_id: str) -> str:
    return f"{TRACE_PREFIX}{trace_id}"


def decode_trace_frame(value: str | None) -> str | None:
    if not value or not value.startswith(TRACE_PREFIX):
        return None
    trace_id = value[len(TRACE_PREFIX):].strip()
    return trace_id or None


def parse_trace_event(data: str) -> dict[str, Any] | None:
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    return event


def publish_trace(
    node: Node,
    publisher,
    trace_id: str | None,
    stage: str,
    event: str,
    *,
    source_stamp: RosTime | None = None,
    **extra: Any,
) -> None:
    if not trace_id:
        return
    payload = {
        "trace_id": trace_id,
        "stage": stage,
        "event": event,
        "t_ros_ns": node_time_ns(node),
        "t_wall_ns": time.time_ns(),
    }
    source_stamp_ns = ros_time_ns(source_stamp)
    if source_stamp_ns is not None:
        payload["source_stamp_ns"] = source_stamp_ns
    payload.update({k: v for k, v in extra.items() if v is not None})
    msg = String()
    msg.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    publisher.publish(msg)

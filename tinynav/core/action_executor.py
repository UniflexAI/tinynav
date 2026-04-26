"""
POI Arrival Action Executor
============================
Runs a sequential list of actions when the robot reaches a POI.
All actions are best-effort: failures are logged and execution continues.

Supported action types:

  lookat   – rotate to face a map-frame coordinate
             params: {"target": [x, y, z]}

  wait     – pause for N seconds
             params: {"seconds": 5}

  photo    – save the latest keyframe image to disk
             params: {"save_path": "/data/photos"}   (optional)

  custom   – publish an arbitrary command string to /service/command
             params: {"command": "bark"}

To add a new action type, implement an async handler with signature
    async def my_handler(params: dict, node) -> None
and register it in ACTION_REGISTRY at the bottom of this file.

Example POI JSON:
  {
    "id": 1,
    "name": "reception",
    "position": [3.2, 1.5, 0.0],
    "actions": [
      {"type": "lookat", "params": {"target": [5.0, 2.0, 0.0]}},
      {"type": "photo",  "params": {}},
      {"type": "wait",   "params": {"seconds": 3}}
    ]
  }
"""
from __future__ import annotations

import asyncio
import math
import os
import time

import cv2
import numpy as np
from std_msgs.msg import String

from tinynav.core.math_utils import np2msg


async def execute_poi_actions(actions: list, node) -> None:
    """Run each action in order. Skips unknown types."""
    for action in actions:
        action_type = action.get("type", "")
        handler = ACTION_REGISTRY.get(action_type)
        if handler is None:
            node.get_logger().warn(f"[action] unknown type '{action_type}', skipping")
            continue
        try:
            await handler(action.get("params", {}), node)
        except Exception as e:  # noqa: BLE001
            node.get_logger().error(f"[action] {action_type} failed: {e}")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_lookat(params: dict, node) -> None:
    """Rotate the robot to face a map-frame target coordinate."""
    target = params.get("target")
    if target is None:
        node.get_logger().warn("[action] lookat: missing 'target' param")
        return
    if node.latest_pose_in_map is None:
        node.get_logger().warn("[action] lookat: no pose available, skipping")
        return

    pose = node.latest_pose_in_map
    dx = target[0] - pose[0, 3]
    dy = target[1] - pose[1, 3]
    desired_yaw = math.atan2(dy, dx)

    # Build a target pose: same position, desired heading only (flat rotation).
    target_mat = np.eye(4)
    target_mat[:3, 3] = pose[:3, 3]
    c, s = math.cos(desired_yaw), math.sin(desired_yaw)
    target_mat[0, 0], target_mat[0, 1] = c, -s
    target_mat[1, 0], target_mat[1, 1] = s,  c

    stamp = node.get_clock().now().to_msg()
    node.target_pose_pub.publish(np2msg(target_mat, stamp, "world", "map"))
    node.get_logger().info(f"[action] lookat target yaw={math.degrees(desired_yaw):.1f}°")

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if node.latest_pose_in_map is not None:
            R = node.latest_pose_in_map[:3, :3]
            current_yaw = math.atan2(R[1, 0], R[0, 0])
            if abs(_angle_diff(current_yaw, desired_yaw)) < math.radians(5):
                node.get_logger().info("[action] lookat: aligned")
                return
        await asyncio.sleep(0.1)

    node.get_logger().warn("[action] lookat: timeout, continuing")


async def _handle_wait(params: dict, node) -> None:
    seconds = float(params.get("seconds", 1))
    node.get_logger().info(f"[action] wait {seconds}s")
    await asyncio.sleep(seconds)


async def _handle_photo(params: dict, node) -> None:
    if node.last_keyframe_image is None:
        node.get_logger().warn("[action] photo: no keyframe image available, skipping")
        return
    save_path = params.get("save_path", "/tmp/tinynav_photos")
    os.makedirs(save_path, exist_ok=True)
    ts = int(time.time())
    fname = os.path.join(save_path, f"poi{node.poi_index}_{ts}.jpg")
    cv2.imwrite(fname, node.last_keyframe_image)
    node.get_logger().info(f"[action] photo saved: {fname}")


async def _handle_custom(params: dict, node) -> None:
    command = params.get("command", "").strip()
    if not command:
        node.get_logger().warn("[action] custom: empty command, skipping")
        return
    node._action_pub.publish(String(data=f"play {command}"))
    node.get_logger().info(f"[action] custom: sent 'play {command}'")


# ---------------------------------------------------------------------------
# Registry — add new action types here
# ---------------------------------------------------------------------------

ACTION_REGISTRY: dict[str, callable] = {
    "lookat": _handle_lookat,
    "wait":   _handle_wait,
    "photo":  _handle_photo,
    "custom": _handle_custom,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d

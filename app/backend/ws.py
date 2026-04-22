"""
WebSocket endpoints:
  WS /ws/status      — pushes device status every ~1 s
  WS /ws/pose        — pushes pose whenever a new Odometry arrives
  WS /ws/map-update  — pushes a notification when map files change
  WS /ws/preview     — streams JPEG frames for a given image topic
  WS /ws/planning    — polls planning snapshot at 5 fps
  WS /ws/teleop      — receives cmd_vel commands from the client
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from .state import runner

router = APIRouter(tags=['ws'])


# --------------------------------------------------------------------------- #
# /ws/status  — polls node state every 1 s and broadcasts                     #
# --------------------------------------------------------------------------- #

@router.websocket('/ws/status')
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            node = runner.node
            if node is not None:
                payload = json.dumps({'online': True, **node.get_status()})
            else:
                payload = json.dumps({'online': False})
            await ws.send_text(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------------------------- #
# /ws/pose  — pushed on every new odometry message                            #
# --------------------------------------------------------------------------- #

@router.websocket('/ws/pose')
async def ws_pose(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)

    loop = asyncio.get_event_loop()

    def _on_pose(pose: dict):
        # Called from rclpy spin thread — schedule safely onto the event loop.
        try:
            loop.call_soon_threadsafe(queue.put_nowait, pose)
        except Exception:
            pass

    node = runner.node
    if node is None:
        await ws.close(code=1013)
        return

    node.pose_callbacks.append(_on_pose)
    try:
        while True:
            pose = await queue.get()
            await ws.send_text(json.dumps(pose))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            node.pose_callbacks.remove(_on_pose)
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# /ws/map-update  — polls for occupancy_grid.npy mtime changes               #
# --------------------------------------------------------------------------- #

@router.websocket('/ws/map-update')
async def ws_map_update(ws: WebSocket):
    await ws.accept()
    node = runner.node
    if node is None:
        await ws.close(code=1013)
        return

    grid_file = os.path.join(node.map_path, 'occupancy_grid.npy')
    last_mtime: float = 0.0

    try:
        while True:
            try:
                mtime = os.path.getmtime(grid_file)
            except OSError:
                mtime = 0.0

            if mtime != last_mtime and mtime != 0.0:
                last_mtime = mtime
                await ws.send_text(json.dumps({
                    'event': 'map_updated',
                    'timestamp': mtime,
                }))
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------------------------- #
# /ws/planning  — polls planning snapshot at 5 fps                            #
# --------------------------------------------------------------------------- #

@router.websocket('/ws/planning')
async def ws_planning(ws: WebSocket):
    await ws.accept()
    node = runner.node
    if node is None:
        await ws.close(code=1013)
        return
    try:
        while True:
            payload = json.dumps(node.get_planning_snapshot())
            await ws.send_text(payload)
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------------------------- #
# /ws/preview  — streams JPEG frames for a given image topic                  #
# --------------------------------------------------------------------------- #

@router.websocket('/ws/preview')
async def ws_preview(ws: WebSocket, topic: str = Query(...)):
    await ws.accept()

    node = runner.node
    if node is None or topic not in node.preview_callbacks:
        await ws.close(code=1013)
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    loop = asyncio.get_event_loop()

    def _on_frame(frame: bytes):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, frame)
        except Exception:
            pass

    node.preview_callbacks[topic].append(_on_frame)
    try:
        while True:
            frame = await queue.get()
            await ws.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            node.preview_callbacks[topic].remove(_on_frame)
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# /ws/teleop  — receives velocity commands and publishes to /cmd_vel          #
# --------------------------------------------------------------------------- #

@router.websocket('/ws/teleop')
async def ws_teleop(ws: WebSocket):
    await ws.accept()
    node = runner.node
    if node is None:
        await ws.close(code=1013)
        return
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            node.publish_cmd_vel(
                float(msg.get('linear_x', 0.0)),
                float(msg.get('linear_y', 0.0)),
                float(msg.get('angular_z', 0.0)),
            )
    except WebSocketDisconnect:
        pass
    finally:
        try:
            node.publish_cmd_vel(0.0, 0.0, 0.0)
        except Exception:
            pass

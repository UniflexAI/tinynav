"""
WebSocket endpoints:
  WS /ws/status      — pushes device status every ~1 s
  WS /ws/pose        — pushes pose whenever a new Odometry arrives
  WS /ws/map-update  — pushes a notification when map files change
"""
from __future__ import annotations

import asyncio
import json
import os
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

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
                payload = json.dumps(node.get_status())
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

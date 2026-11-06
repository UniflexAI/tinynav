"""
Mock backend for Flutter app UI testing — no ROS2 required.

Run on HOST:
    uv run --with fastapi --with uvicorn app/backend/mock_server.py
or:
    pip install fastapi uvicorn && python app/backend/mock_server.py
"""
from __future__ import annotations

import asyncio
import json
import math
import struct
import time
import zlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ── Simulated state ──────────────────────────────────────────────────────────

_state: dict = {
    'bag_recording': False,
    'bag_file_ready': True,
    'map_built': True,
    'mapping_percent': 0.0,
    'navigating': False,
    'raw_state': 'idle',
}

_pois: dict[str, dict] = {
    '0': {'id': 0, 'name': 'Entrance', 'position': [1.0, 2.0, 0.0]},
    '1': {'id': 1, 'name': 'Office',   'position': [3.5, 1.5, 0.0]},
}


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title='TinyNav Mock API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── Device ───────────────────────────────────────────────────────────────────

@app.get('/device/info')
def device_info():
    return {
        'deviceId': 'mock-device',
        'firmwareVersion': '0.1.0-mock',
        'capabilities': ['bag_record', 'map_build', 'navigation'],
    }


@app.get('/device/status')
def device_status():
    return _build_status()


def _build_status() -> dict:
    s = _state
    return {
        'online': True,
        'battery': 0.85,
        'bagStatus': 'recording' if s['bag_recording'] else 'idle',
        'bagFileReady': s['bag_file_ready'],
        'mapStatus': (
            'building' if s['raw_state'] == 'rosbag_build_map' else
            'success'  if s['map_built'] else 'idle'
        ),
        'mappingPercent': s['mapping_percent'],
        'navStatus': 'navigating' if s['navigating'] else 'idle',
        'rawState': s['raw_state'],
    }


# ── Bag ──────────────────────────────────────────────────────────────────────

@app.post('/bag/start')
def bag_start():
    if _state['raw_state'] != 'idle':
        raise HTTPException(409, f"Cannot start bag while in state: {_state['raw_state']}")
    _state['bag_recording'] = True
    _state['raw_state'] = 'realsense_bag_record'
    return {'ok': True}


@app.post('/bag/stop')
def bag_stop():
    if not _state['bag_recording']:
        raise HTTPException(409, 'Not recording')
    _state['bag_recording'] = False
    _state['bag_file_ready'] = True
    _state['raw_state'] = 'idle'
    return {'ok': True}


@app.get('/bag/status')
def bag_status():
    return {
        'status': 'recording' if _state['bag_recording'] else 'idle',
        'bagFileReady': _state['bag_file_ready'],
    }


# ── Map ──────────────────────────────────────────────────────────────────────

@app.post('/map/build')
async def map_build():
    if not _state['bag_file_ready']:
        raise HTTPException(400, 'No bag file — record a bag first')
    if _state['raw_state'] != 'idle':
        raise HTTPException(409, f"Cannot build map while in state: {_state['raw_state']}")
    _state['raw_state'] = 'rosbag_build_map'
    _state['map_built'] = False
    _state['mapping_percent'] = 0.0
    asyncio.create_task(_simulate_map_build())
    return {'ok': True}


async def _simulate_map_build():
    for i in range(1, 11):
        await asyncio.sleep(1.0)
        _state['mapping_percent'] = float(i * 10)
    _state['map_built'] = True
    _state['raw_state'] = 'idle'


@app.get('/map/current')
def map_current():
    if not _state['map_built']:
        raise HTTPException(404, 'No map available')
    return {
        'imageUrl': '/map/image',
        'origin_x': -5.0,
        'origin_y': -5.0,
        'resolution': 0.05,
        'width': 200,
        'height': 200,
    }


@app.get('/map/image', response_class=Response)
def map_image():
    """Return a simple synthetic PNG (200x200): gray background + room outline."""
    w, h = 200, 200
    rows: list[bytes] = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            # Wall border
            if x in (40, 160) or y in (40, 160):
                v = 35
            # Free space inside
            elif 40 < x < 160 and 40 < y < 160:
                v = 210
            else:
                v = 128
            row += bytes([v, v, v])
        rows.append(bytes([0]) + bytes(row))  # filter byte + RGB

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFF_FFFF
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)

    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    png = (
        b'\x89PNG\r\n\x1a\n'
        + _chunk(b'IHDR', ihdr)
        + _chunk(b'IDAT', zlib.compress(b''.join(rows)))
        + _chunk(b'IEND', b'')
    )
    return Response(content=png, media_type='image/png')


# ── POI ──────────────────────────────────────────────────────────────────────

@app.get('/map/pois')
def list_pois():
    return {'pois': list(_pois.values())}


@app.post('/map/pois')
def create_poi(req: dict):
    new_id = max((int(k) for k in _pois), default=-1) + 1
    _pois[str(new_id)] = {'id': new_id, 'name': req['name'], 'position': req['position']}
    return _pois[str(new_id)]


@app.delete('/poi/{poi_id}')
def delete_poi(poi_id: int):
    if str(poi_id) not in _pois:
        raise HTTPException(404, f'POI {poi_id} not found')
    del _pois[str(poi_id)]
    return {'ok': True}


# ── Navigation ───────────────────────────────────────────────────────────────

@app.post('/nav/go-to-poi')
async def nav_go(req: dict):
    if _state['raw_state'] != 'idle':
        raise HTTPException(409, f"Cannot navigate while in state: {_state['raw_state']}")
    _state['navigating'] = True
    _state['raw_state'] = 'navigation'
    asyncio.create_task(_simulate_nav())
    return {'ok': True, 'poi_id': req['poi_id']}


async def _simulate_nav():
    await asyncio.sleep(8.0)
    _state['navigating'] = False
    _state['raw_state'] = 'idle'


@app.post('/nav/cancel')
def nav_cancel():
    if not _state['navigating']:
        raise HTTPException(409, 'Not navigating')
    _state['navigating'] = False
    _state['raw_state'] = 'idle'
    return {'ok': True}


@app.get('/nav/status')
def nav_status():
    return {
        'status': 'navigating' if _state['navigating'] else 'idle',
        'rawState': _state['raw_state'],
    }


# ── WebSockets ───────────────────────────────────────────────────────────────

@app.websocket('/ws/status')
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(_build_status()))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


@app.websocket('/ws/pose')
async def ws_pose(ws: WebSocket):
    """Simulate robot moving in a slow circle around map centre."""
    await ws.accept()
    t = 0.0
    try:
        while True:
            x = 2.0 * math.cos(t)
            y = 2.0 * math.sin(t)
            yaw = t + math.pi / 2
            await ws.send_text(json.dumps({'x': x, 'y': y, 'yaw': yaw, 'timestamp': time.time()}))
            t += 0.05
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.websocket('/ws/map-update')
async def ws_map_update(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(30.0)
    except WebSocketDisconnect:
        pass


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)

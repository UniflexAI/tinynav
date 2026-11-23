"""Isolated App profile for the loop-owned MuJoCo/TinyNav development runtime."""
import asyncio
import anyio
import json
import os
import threading
from pathlib import Path
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title='TinyNav MuJoCo development API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
CONTROL = 'http://127.0.0.1:' + os.environ.get('TINYNAV_SIM_CONTROL_PORT', '8013')


def call(path, payload=None, *, image=False):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(CONTROL + path, data=body,
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
            return raw if image else json.loads(raw)
    except urllib.error.HTTPError as exc:
        raise HTTPException(exc.code, json.loads(exc.read()).get('detail', str(exc))) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(503, 'MuJoCo runtime unavailable') from exc


class Task(BaseModel):
    model_config = {"extra": "forbid"}
    direction: Literal['up', 'down'] = 'down'
    floors: int = Field(9, ge=1, le=9)
    landing_depth: Literal[1.2, 1.75] = 1.2
    noise_mode: Literal['none', 'position', 'yaw', 'pitch', 'combined'] = 'combined'
    noise_scale: Literal[0.5, 1, 2] = 0.5
    noise_seed: Literal[7, 17, 27] = 7
    speed: Literal[1, 2, 4] = 1


class Camera(BaseModel):
    mode: Literal['Free orbit', 'Follow robot', 'Overview']


@app.get('/device/info')
def device_info():
    capabilities=['simulation','stair','manual_control']
    return dict(deviceId='MuJoCo · Go2 · simulated', firmwareVersion='stair-dev', capabilities=capabilities,
                stair_support=dict(directions=['up','down'],min_floors=1,max_floors=9,landings_per_floor=2,camera_height=.4669))


@app.get('/simulation')
def simulation():
    return dict(enabled=True, viewer_port=int(os.environ.get('TINYNAV_SIM_PUBLIC_VIEWER_PORT', '18082')),
                execution='stair_node → planning_node → cmd_vel_control → MuJoCo Go2 policy',
                transport='in-process deterministic simulation clock; no physical robot')


@app.get('/simulation/state')
def state():
    return call('/state')


@app.post('/simulation/start')
def start(task: Task):
    return call('/start', task.model_dump())


@app.post('/simulation/reset')
def reset(task: Task):
    return call('/reset', task.model_dump())


@app.post('/simulation/pause')
@app.post('/nav/pause')
def pause():
    return call('/pause', {})


@app.post('/simulation/resume')
@app.post('/nav/resume')
def resume():
    return call('/resume', {})


@app.post('/simulation/stop')
@app.post('/nav/stair/stop')
def stop():
    return call('/stop', {})


@app.post('/simulation/camera')
def camera(req: Camera):
    return call('/camera', req.model_dump())


@app.get('/simulation/images/{name}')
def frame(name: Literal['planner', 'depth', 'depth-high', 'color', 'color-high']):
    return Response(call('/images/' + name, image=True), media_type='image/jpeg', headers={'Cache-Control': 'no-store'})


def status():
    s = state()
    running = s.get('ready', False) and not (s.get('cancelled') or s.get('validation_passed') or s.get('validation_error')) and s.get('status') != 'Ready' and s.get('task_active',True)
    return dict(online=s.get('ready', False), rawState='stair' if running else 'idle',
                navNodesRunning=s.get('ready', False) and not s.get('cancelled', False), navPaused=s.get('paused', True),
                stairMode=s.get('direction') if running else None, stairStatus=s.get('status', ''),
                vioSafety=s.get('vio_safety', {}),
                mapStatus='idle')


class VioConfirmation(BaseModel):
    fault_id: int = Field(ge=1, strict=True)


class VioFaultRequest(BaseModel):
    kind: Literal['jump', 'dropout', 'restart']
    duration: float = Field(2., ge=.5, le=10.)


@app.get('/nav/vio')
def vio_status():
    return state().get('vio_safety', {})


@app.post('/nav/vio/confirm')
def vio_confirm(req: VioConfirmation):
    return call('/vio_confirm', req.model_dump())


@app.post('/simulation/vio-fault')
def vio_fault(req: VioFaultRequest):
    return call('/fault', req.model_dump())


@app.get('/device/status')
@app.get('/nav/status')
def device_status():
    return status()


@app.websocket('/ws/status')
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            try:
                payload = await asyncio.to_thread(status)
            except HTTPException:
                payload = {'online': False}
            await ws.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

# Original TinyNav App routes and WebSockets. The MuJoCo page remains a separate
# observer; these carry the original App's structured planner/camera contracts.
_SETTINGS_PATH = Path(os.environ.get('TINYNAV_SIM_STAIR_SETTINGS', '/devwork/mujoco-app-stair-settings.json'))
_SETTINGS_LOCK = threading.Lock()
_COLOR = '/camera/camera/color/image_raw'
_DEPTH = '/slam/depth'


class AppStairSettings(BaseModel):
    floors: int = Field(9, ge=1, le=9)
    landings_per_floor: Literal[2] = 2
    camera_height: float | None = Field(None, ge=0.2, le=1.2)


class AppStairTask(BaseModel):
    direction: Literal['up', 'down']
    turn: Literal['auto'] = 'auto'
    floors: int | None = Field(None, ge=1, le=9)
    landings_per_floor: Literal[2] = 2


@app.get('/nav/stair/settings')
def stair_settings():
    with _SETTINGS_LOCK:
        if _SETTINGS_PATH.exists():
            return AppStairSettings(**json.loads(_SETTINGS_PATH.read_text())).model_dump()
    return dict(floors=9, landings_per_floor=2, camera_height=0.4669)


@app.put('/nav/stair/settings')
def save_stair_settings(req: AppStairSettings):
    if req.camera_height is not None and abs(req.camera_height - 0.4669) > 0.005:
        raise HTTPException(409, 'MuJoCo camera calibration is fixed at 0.4669 m')
    settings = {**req.model_dump(), 'camera_height': 0.4669}
    with _SETTINGS_LOCK:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(json.dumps(settings))
    return settings


def applied_config():
    s = state()
    noise = s.get('odom_noise', {})
    return dict(direction=s.get('direction', 'down'), floors=s.get('floors', 9),
                landing_depth=s.get('landing_depth', 1.2), speed=s.get('speed', 1),
                noise_mode=noise.get('mode', 'combined'), noise_scale=noise.get('scale', 0.5),
                noise_seed=noise.get('seed', 7))


@app.post('/nav/stair/start')
def app_stair_start(req: AppStairTask):
    cfg = applied_config()
    cfg.update(direction=req.direction, floors=req.floors or stair_settings()['floors'])
    result = call('/start', cfg)
    return {'ok': True, 'direction': req.direction, 'turn': req.turn, 'floors': cfg['floors'],
            'landings_per_floor': 2, 'run_id': result['run_id']}


@app.post('/nav/nodes/disable')
def disable_nodes():
    return stop()


@app.post('/nav/nodes/enable')
def enable_nodes():
    return call('/reset', applied_config())


@app.get('/sensor/mode')
def sensor_mode():
    return {'mode': 'mujoco'}


@app.get('/sensor/image-topics')
def image_topics():
    return {'topics': [_COLOR, _DEPTH]}


@app.get('/map/pois')
def empty_pois():
    return {'pois': []}


@app.get('/map/current')
def no_global_map():
    raise HTTPException(404,'Stair simulation uses live local perception; no global map')


@app.get('/simulation/planning')
def planning():
    return call('/planning')


@app.websocket('/ws/planning')
async def ws_planning(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(await asyncio.to_thread(planning))
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.websocket('/ws/pose')
async def ws_pose(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            snapshot = await asyncio.to_thread(planning)
            if snapshot.get('odom_pose'):
                await ws.send_json(snapshot['odom_pose'])
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.websocket('/ws/preview')
async def ws_preview(ws: WebSocket, topic: str = Query(...), quality: str = Query('default')):
    await ws.accept()
    if topic not in (_COLOR, _DEPTH) or quality not in ('default', 'high'):
        await ws.close(code=1008, reason='Unsupported simulated image topic/profile')
        return
    image_name = ('color' if topic == _COLOR else 'depth') + ('-high' if quality == 'high' else '')
    try:
        while True:
            try:
                frame = await asyncio.to_thread(call, '/images/' + image_name, image=True)
                await ws.send_bytes(frame)
            except HTTPException as exc:
                if exc.status_code not in (404, 503):
                    raise
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.websocket('/ws/teleop')
async def sim_teleop(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            command=await ws.receive_json()
            await asyncio.to_thread(call,'/teleop',command)
    except WebSocketDisconnect:
        pass
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(call,'/teleop',dict(linear_x=0.,linear_y=0.,angular_z=0.))

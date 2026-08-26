import os
import socket

from fastapi import APIRouter

from ..state import runner

router = APIRouter(tags=['sensor'])


@router.get('/sensor/mode')
def get_sensor_mode():
    node = runner.node
    return {'mode': node.get_sensor_mode() if node else 'unknown'}


@router.get('/sensor/image-topics')
def get_image_topics():
    node = runner.node
    return {'topics': node.get_image_topics() if node else []}


@router.get('/sensor/hikvision-status')
def hikvision_status():
    """Quick reachability probe for the external Hikvision gimbal camera: a bare
    TCP connect to its RTSP port, no RTSP handshake or frame grab. Lets a caller
    (the web app's "start mission" button) check before dispatching whether
    CaptureHikShot is likely to succeed, instead of finding out only after a
    whole mission has run with nothing landing in poi_shots_hik.

    Uses the same HIK_CAMERA_* config as CaptureHikShot
    (core_runtime/mission/leaves.py) — see that module's _camera_config for why
    _load_env_file() has to run first (HIK_CAMERA_* lives in tinynav/.env,
    nothing sources it into this process's environment on its own)."""
    import tool.hikvision_snapshot as hik

    hik._load_env_file()
    ip = os.environ.get('HIK_CAMERA_IP')
    port_raw = os.environ.get('HIK_CAMERA_PORT', '554')
    if not ip:
        return {'reachable': False, 'reason': 'not_configured'}
    port = int(port_raw)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect((ip, port))
        return {'reachable': True, 'ip': ip, 'port': port}
    except OSError as exc:
        return {'reachable': False, 'ip': ip, 'port': port, 'reason': str(exc)}
    finally:
        s.close()

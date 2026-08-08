import socket

import psutil
from fastapi import APIRouter, HTTPException

from ..rtk_tmux import restart_rtk_bridge
from ..state import runner

router = APIRouter(tags=['device'])


@router.get('/info')
def device_info():
    return {
        'deviceId': socket.gethostname(),
        'firmwareVersion': '0.1.0',
        'capabilities': ['bag_record', 'map_build', 'navigation'],
    }


@router.get('/status')
def device_status():
    node = runner.node
    if node is None:
        return {'online': False, 'battery': None, **_empty_status()}
    status = node.get_status()
    return {
        'online': True,
        'battery': None,   # no battery topic yet
        **status,
    }


@router.get('/sysinfo')
def device_sysinfo():
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return {
        'cpu_percent': round(cpu, 1),
        'mem_percent': round(mem.percent, 1),
        'mem_used_gb': round(mem.used / 1024 ** 3, 1),
        'mem_total_gb': round(mem.total / 1024 ** 3, 1),
        'disk_percent': round(disk.percent, 1),
        'disk_used_gb': round(disk.used / 1024 ** 3, 1),
        'disk_total_gb': round(disk.total / 1024 ** 3, 1),
        'gpu_percent': _jetson_gpu_percent(),
    }


@router.post('/rtk/restart')
def restart_rtk():
    """Kill and re-launch scripts/run_rtk.sh in the app tmux RTK pane."""
    try:
        pane = restart_rtk_bridge()
    except RuntimeError as exc:
        message = str(exc)
        status = 504 if 'timed out' in message else 500
        raise HTTPException(status, message) from exc
    return {'ok': True, 'pane': pane}


def _jetson_gpu_percent() -> float | None:
    # Jetson: /sys/devices/gpu.0/load reports 0–1000 (tenths of a percent)
    try:
        with open('/sys/devices/gpu.0/load') as f:
            return round(int(f.read().strip()) / 10.0, 1)
    except Exception:
        return None


def _empty_status():
    return {
        'bagStatus': 'idle',
        'bagFileReady': False,
        'mapStatus': 'idle',
        'mappingPercent': 0.0,
        'navStatus': 'idle',
        'rawState': 'unknown',
    }

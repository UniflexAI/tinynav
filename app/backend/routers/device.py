from fastapi import APIRouter
from ..state import runner
import socket

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


def _empty_status():
    return {
        'bagStatus': 'idle',
        'bagFileReady': False,
        'mapStatus': 'idle',
        'mappingPercent': 0.0,
        'navStatus': 'idle',
        'rawState': 'unknown',
    }

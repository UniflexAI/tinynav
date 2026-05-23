from fastapi import APIRouter, HTTPException
from ..state import runner

router = APIRouter(tags=['debug_record'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


@router.post('/start')
def debug_record_start():
    node = _require_node()
    if node.debug_recording:
        raise HTTPException(409, 'Already recording')
    node.cmd_debug_record_start()
    return {'ok': True}


@router.post('/stop')
def debug_record_stop():
    node = _require_node()
    if not node.debug_recording:
        raise HTTPException(409, 'Not recording')
    node.cmd_debug_record_stop()
    return {'ok': True}


@router.get('/status')
def debug_record_status():
    node = _require_node()
    return {
        'recording': node.debug_recording,
        'path': node.debug_record_path,
    }

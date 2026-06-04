from fastapi import APIRouter, HTTPException
from ..state import runner

router = APIRouter(tags=['bag'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


@router.post('/start')
def bag_start():
    node = _require_node()
    node.recover_stale_error_state()
    if node.is_bag_recording():
        raise HTTPException(409, 'Already recording')
    if node.state not in ('idle',):
        raise HTTPException(409, f'Cannot start bag while in state: {node.state}')
    node.cmd_bag_start()
    return {'ok': True}


@router.post('/stop')
def bag_stop():
    node = _require_node()
    node.recover_stale_error_state()
    if not node.is_bag_recording() and node.state != 'realsense_bag_record':
        raise HTTPException(409, 'Not recording')
    if not node.cmd_bag_stop():
        raise HTTPException(500, 'Failed to stop bag recorder')
    return {'ok': True}


@router.get('/status')
def bag_status():
    node = _require_node()
    return node.get_bag_status()

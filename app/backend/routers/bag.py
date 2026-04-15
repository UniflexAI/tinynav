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
    if node.state == 'realsense_bag_record':
        raise HTTPException(409, 'Already recording')
    if node.state not in ('idle',):
        raise HTTPException(409, f'Cannot start bag while in state: {node.state}')
    node.cmd_bag_start()
    return {'ok': True}


@router.post('/stop')
def bag_stop():
    node = _require_node()
    if node.state != 'realsense_bag_record':
        raise HTTPException(409, 'Not recording')
    node.cmd_bag_stop()
    return {'ok': True}


@router.get('/status')
def bag_status():
    node = _require_node()
    import os
    bag_file = os.path.join(node.bag_path, 'bag_0.db3')
    return {
        'status': 'recording' if node.state == 'realsense_bag_record' else 'idle',
        'bagFileReady': os.path.exists(bag_file),
        'bagPath': node.bag_path,
    }

from fastapi import APIRouter, HTTPException

from ..state import runner

router = APIRouter(tags=['field_record'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


@router.get('/info')
def field_record_info():
    node = _require_node()
    recording, path, seconds = node.field_record_state()
    return {'recording': recording, 'path': path, 'seconds': seconds, 'topics': node.FIELD_RECORD_TOPICS}


@router.post('/start')
def field_record_start():
    node = _require_node()
    if node.field_record_state()[0]:
        raise HTTPException(409, 'Already recording')
    node.cmd_field_record_start()
    return {'ok': True, 'path': node.field_record_state()[1]}


@router.post('/stop')
def field_record_stop():
    node = _require_node()
    recording, path, _ = node.field_record_state()
    if not recording:
        raise HTTPException(409, 'Not recording')
    node.cmd_field_record_stop()
    return {'ok': True, 'path': path}

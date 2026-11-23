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
    return node.field_record_summary()


@router.post('/start')
def field_record_start():
    node = _require_node()
    if node.field_record_state()[0]:
        raise HTTPException(409, 'Already recording')
    try:
        node.cmd_field_record_start()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return {'ok': True, **node.field_record_summary()}


@router.post('/stop')
def field_record_stop():
    node = _require_node()
    recording, path, _ = node.field_record_state()
    if not recording:
        raise HTTPException(409, 'Not recording')
    try:
        node.cmd_field_record_stop()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {'ok': True, **node.field_record_summary()}

from fastapi import APIRouter, Body, HTTPException

from ..state import runner

router = APIRouter(tags=['benchmark'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


@router.post('/start')
def benchmark_start(payload: dict | None = Body(default=None)):
    node = _require_node()
    if not node.has_odom:
        raise HTTPException(409, 'Odometry not ready')
    node.cmd_start_benchmark(payload or {})
    return {'ok': True}


@router.post('/stop')
def benchmark_stop():
    node = _require_node()
    node.cmd_stop_benchmark()
    return {'ok': True}


@router.post('/restart')
def benchmark_restart(payload: dict | None = Body(default=None)):
    node = _require_node()
    if not node.has_odom:
        raise HTTPException(409, 'Odometry not ready')
    node.cmd_restart_benchmark(payload or {})
    return {'ok': True}


@router.get('/status')
def benchmark_status():
    node = _require_node()
    return node.get_benchmark_status()

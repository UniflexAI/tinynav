from fastapi import APIRouter, HTTPException

from ..state import runner

router = APIRouter(tags=['benchmark'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


@router.post('/start')
def benchmark_start():
    node = _require_node()
    if node._odom_pose is None:
        raise HTTPException(409, 'Odometry not ready')
    node.cmd_start_benchmark()
    return {'ok': True}


@router.post('/stop')
def benchmark_stop():
    node = _require_node()
    node.cmd_stop_benchmark()
    return {'ok': True}


@router.post('/restart')
def benchmark_restart():
    node = _require_node()
    if node._odom_pose is None:
        raise HTTPException(409, 'Odometry not ready')
    node.cmd_restart_benchmark()
    return {'ok': True}


@router.get('/status')
def benchmark_status():
    node = _require_node()
    return node.get_benchmark_status()

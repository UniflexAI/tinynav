from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import runner

router = APIRouter(tags=['nav'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


class GoToPoiRequest(BaseModel):
    poi_id: int


@router.post('/go-to-poi')
def nav_go_to_poi(req: GoToPoiRequest):
    node = _require_node()
    if node.state == 'navigation':
        raise HTTPException(409, 'Already navigating')
    if node.state not in ('idle',):
        raise HTTPException(409, f'Cannot start navigation while in state: {node.state}')
    node.cmd_nav_start(poi_id=str(req.poi_id))
    return {'ok': True, 'poi_id': req.poi_id}


@router.post('/cancel')
def nav_cancel():
    node = _require_node()
    if node.state != 'navigation':
        raise HTTPException(409, 'Not navigating')
    node.cmd_nav_cancel()
    return {'ok': True}


@router.get('/status')
def nav_status():
    node = _require_node()
    return {
        'status': 'navigating' if node.state == 'navigation' else 'idle',
        'rawState': node.state,
    }


@router.post('/nodes/enable')
def nav_nodes_enable():
    node = _require_node()
    if node._nav_nodes_running:
        raise HTTPException(409, 'Nav nodes already running')
    node.cmd_start_nav_nodes()
    return {'ok': True}


@router.post('/restart')
def nav_restart():
    node = _require_node()
    if not node._nav_nodes_running:
        raise HTTPException(409, 'Nav nodes not running')
    node.cmd_restart_nav_nodes()
    return {'ok': True}


@router.post('/nodes/disable')
def nav_nodes_disable():
    node = _require_node()
    if not node._nav_nodes_running:
        raise HTTPException(409, 'Nav nodes not running')
    node.cmd_stop_nav_nodes()
    return {'ok': True}

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


class SendPoisRequest(BaseModel):
    poi_ids: list[int]


class ManualTargetRequest(BaseModel):
    x: float
    y: float
    z: float
    activate: bool = True


@router.post('/send-pois')
def nav_send_pois(req: SendPoisRequest):
    node = _require_node()
    if not node._localized:
        raise HTTPException(409, 'Not localized')
    node.cmd_send_pois(req.poi_ids)
    return {'ok': True}


@router.post('/go-to-poi')
def nav_go_to_poi(req: GoToPoiRequest):
    node = _require_node()
    if node.state == 'navigation':
        raise HTTPException(409, 'Already navigating')
    if node.state not in ('idle',):
        raise HTTPException(409, f'Cannot start navigation while in state: {node.state}')
    node.cmd_nav_start(poi_id=str(req.poi_id))
    return {'ok': True, 'poi_id': req.poi_id}


@router.post('/manual-target')
def nav_manual_target(req: ManualTargetRequest):
    node = _require_node()
    if node._odom_pose is None:
        raise HTTPException(409, 'Odometry not ready')
    if node.state == 'navigation' and not node._manual_target_active:
        raise HTTPException(409, 'Manual target is disabled while POI navigation is running')
    if req.activate and not node._nav_nodes_running:
        raise HTTPException(409, 'Nav nodes not running')
    node.cmd_manual_target_pose(req.x, req.y, req.z, activate=req.activate)
    return {
        'ok': True,
        'target': {'x': req.x, 'y': req.y, 'z': req.z},
        'active': req.activate,
    }


@router.post('/manual-target/clear')
def nav_manual_target_clear():
    node = _require_node()
    node.cmd_clear_manual_target_pose()
    return {'ok': True}


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


@router.post('/pause')
def nav_pause():
    node = _require_node()
    node.cmd_nav_pause()
    return {'ok': True}


@router.post('/resume')
def nav_resume():
    node = _require_node()
    node.cmd_nav_resume()
    return {'ok': True}


@router.post('/nodes/disable')
def nav_nodes_disable():
    node = _require_node()
    if not node._nav_nodes_running:
        raise HTTPException(409, 'Nav nodes not running')
    node.cmd_stop_nav_nodes()
    return {'ok': True}

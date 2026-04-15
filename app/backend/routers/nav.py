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

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import runner

router = APIRouter(tags=['bag'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


class PoiMarkRequest(BaseModel):
    name: str
    timestamp_ns: Optional[int] = None


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
        'poiMarkCount': node.get_poi_mark_count(),
    }


@router.post('/poi-marks')
def bag_poi_mark(req: PoiMarkRequest):
    node = _require_node()
    if node.state != 'realsense_bag_record':
        raise HTTPException(409, 'POI marks can only be recorded while bag recording')
    try:
        mark = node.record_poi_mark(req.name, req.timestamp_ns)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {'ok': True, 'mark': mark, 'count': node.get_poi_mark_count()}

"""
POI management — reads/writes pois.json in the map directory.

pois.json schema:
  { "<id_str>": {"id": int, "name": str, "position": [x, y, z]} }
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import runner

router = APIRouter(tags=['poi'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


def _pois_path(node) -> str:
    return os.path.join(node.map_path, 'pois.json')


def _load_pois(node) -> dict:
    path = _pois_path(node)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_pois(node, pois: dict):
    with open(_pois_path(node), 'w') as f:
        json.dump(pois, f, indent=2)


class PoiCreateRequest(BaseModel):
    name: str
    position: list[float]   # [x, y, z]


@router.get('/map/pois')
def list_pois():
    node = _require_node()
    pois = _load_pois(node)
    return {'pois': list(pois.values())}


@router.post('/map/pois')
def create_poi(req: PoiCreateRequest):
    node = _require_node()
    if len(req.position) != 3:
        raise HTTPException(400, 'position must be [x, y, z]')
    pois = _load_pois(node)

    # Use next integer ID
    existing_ids = [int(k) for k in pois.keys()] if pois else []
    new_id = max(existing_ids) + 1 if existing_ids else 0

    pois[str(new_id)] = {
        'id': new_id,
        'name': req.name,
        'position': req.position,
    }
    _save_pois(node, pois)
    return pois[str(new_id)]


@router.delete('/poi/{poi_id}')
def delete_poi(poi_id: int):
    node = _require_node()
    pois = _load_pois(node)
    key = str(poi_id)
    if key not in pois:
        raise HTTPException(404, f'POI {poi_id} not found')
    del pois[key]
    _save_pois(node, pois)
    return {'ok': True}

import io
import json
import os
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from ..map_renderer import render_map
from ..state import runner

router = APIRouter(tags=['map'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


class MapBuildRequest(BaseModel):
    bag_name: Optional[str] = None


@router.post('/build')
def map_build(req: MapBuildRequest = MapBuildRequest()):
    node = _require_node()
    if req.bag_name:
        node.set_active_bag(req.bag_name)
    active_bag = node.active_bag_path
    if active_bag is None or not os.path.exists(os.path.join(active_bag, 'bag_0.db3')):
        raise HTTPException(400, 'No verified bag available — select a bag or record a new one')
    if node.state == 'rosbag_build_map':
        raise HTTPException(409, 'Already building map')
    if node.state not in ('idle',):
        raise HTTPException(409, f'Cannot build map while in state: {node.state}')
    node.cmd_map_build()
    return {'ok': True}


@router.get('/current')
def map_current():
    """Returns map metadata + image URL. Image served at /map/image."""
    node = _require_node()
    grid_file = os.path.join(node.map_path, 'occupancy_grid.npy')
    if not os.path.exists(grid_file):
        raise HTTPException(404, 'No map available')
    try:
        _, meta = render_map(node.map_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {
        'imageUrl': '/map/image',
        **meta,
    }


@router.get('/image', response_class=Response)
def map_image():
    """Returns the occupancy grid as a PNG image."""
    node = _require_node()
    try:
        png_bytes, _ = render_map(node.map_path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(content=png_bytes, media_type='image/png')


@router.post('/set-active/{map_name}')
def map_set_active(map_name: str):
    """Symlink tinynav_db/map → maps/{map_name}, replacing any existing map link."""
    import shutil
    if not re.match(r'^[a-zA-Z0-9_\-]+$', map_name):
        raise HTTPException(400, 'Invalid map name')
    root = os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')
    src = os.path.join(root, 'maps', map_name)
    if not os.path.isdir(src):
        raise HTTPException(404, f'Map {map_name!r} not found')
    link = os.path.join(root, 'map')
    if os.path.islink(link) or os.path.isfile(link):
        os.remove(link)
    elif os.path.isdir(link):
        shutil.rmtree(link)
    os.symlink(src, link)
    return {'ok': True, 'active': map_name}


def _resolve_map_path(map_name: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_\-]+$', map_name):
        raise HTTPException(400, 'Invalid map name')
    root = os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')
    path = os.path.join(root, 'maps', map_name)
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, 'occupancy_grid.npy')):
        raise HTTPException(404, f'Map {map_name!r} not found')
    return path


@router.get('/preview/{map_name}')
def map_preview_info(map_name: str):
    """Metadata + POIs for a named map folder."""
    path = _resolve_map_path(map_name)
    try:
        png_bytes, meta = render_map(path)
    except Exception as e:
        raise HTTPException(500, str(e))

    img = Image.open(io.BytesIO(png_bytes))
    img_w, img_h = img.size  # PIL (width, height)

    pois: list = []
    pois_file = os.path.join(path, 'pois.json')
    if os.path.exists(pois_file):
        with open(pois_file) as f:
            pois = list(json.load(f).values())

    return {
        'imageUrl': f'/map/preview/{map_name}/image',
        'origin_x': meta['origin_x'],
        'origin_y': meta['origin_y'],
        'resolution': meta['resolution'],
        'width': img_w,
        'height': img_h,
        'pois': pois,
    }


@router.get('/preview/{map_name}/image', response_class=Response)
def map_preview_image(map_name: str):
    """Rendered PNG for a named map folder."""
    path = _resolve_map_path(map_name)
    try:
        png_bytes, _ = render_map(path)
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(content=png_bytes, media_type='image/png')


class MapPoiCreateRequest(BaseModel):
    name: str
    position: list[float]  # [x, y, z]


@router.post('/preview/{map_name}/pois')
def map_preview_create_poi(map_name: str, req: MapPoiCreateRequest):
    path = _resolve_map_path(map_name)
    if len(req.position) != 3:
        raise HTTPException(400, 'position must be [x, y, z]')
    pois_file = os.path.join(path, 'pois.json')
    pois: dict = {}
    if os.path.exists(pois_file):
        with open(pois_file) as f:
            pois = json.load(f)
    existing_ids = [int(k) for k in pois.keys()] if pois else []
    new_id = max(existing_ids) + 1 if existing_ids else 0
    pois[str(new_id)] = {'id': new_id, 'name': req.name, 'position': req.position}
    with open(pois_file, 'w') as f:
        json.dump(pois, f, indent=2)
    return pois[str(new_id)]


@router.delete('/preview/{map_name}/pois/{poi_id}')
def map_preview_delete_poi(map_name: str, poi_id: int):
    path = _resolve_map_path(map_name)
    pois_file = os.path.join(path, 'pois.json')
    if not os.path.exists(pois_file):
        raise HTTPException(404, f'POI {poi_id} not found')
    with open(pois_file) as f:
        pois = json.load(f)
    key = str(poi_id)
    if key not in pois:
        raise HTTPException(404, f'POI {poi_id} not found')
    del pois[key]
    with open(pois_file, 'w') as f:
        json.dump(pois, f, indent=2)
    return {'ok': True}

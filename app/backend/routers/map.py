import io
import json
import os
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image

from ..map_renderer import render_map
from ..state import runner

router = APIRouter(tags=['map'])


def _require_node():
    if runner.node is None:
        raise HTTPException(503, 'ROS node not ready')
    return runner.node


@router.post('/build')
def map_build():
    node = _require_node()
    bag_file = os.path.join(node.bag_path, 'bag_0.db3')
    if not os.path.exists(bag_file):
        raise HTTPException(400, 'No bag file found — record a bag first')
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


def _resolve_map_path(map_name: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_\-]+$', map_name):
        raise HTTPException(400, 'Invalid map name')
    root = os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')
    path = os.path.join(root, map_name)
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

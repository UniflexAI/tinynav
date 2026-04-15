import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

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

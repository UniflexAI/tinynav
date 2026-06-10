from fastapi import APIRouter
from ..state import runner

router = APIRouter(tags=['sensor'])


@router.get('/sensor/mode')
def get_sensor_mode():
    node = runner.node
    return {'mode': node.get_sensor_mode() if node else 'unknown'}


@router.get('/sensor/image-topics')
def get_image_topics():
    node = runner.node
    return {'topics': node.get_image_topics() if node else []}

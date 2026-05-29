import os
import json
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix='/files', tags=['files'])


def _db_root() -> Path:
    default_root = Path(__file__).resolve().parents[3] / 'tinynav_db'
    return Path(os.environ.get('TINYNAV_DB_PATH', str(default_root)))


def _path_size(p: Path) -> int:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    return p.stat().st_size


def _list_dir(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = sorted(path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            'name': p.name,
            'size': _path_size(p),
            'mtime': p.stat().st_mtime,
            'is_dir': p.is_dir(),
        }
        for p in entries
    ]


def _bag_meta_file() -> Path:
    return _db_root() / 'rosbags' / '.descriptors.json'


def _load_bag_descriptors() -> dict[str, str]:
    meta_path = _bag_meta_file()
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text())
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except Exception:
        return {}
    return {}


def _save_bag_descriptors(meta: dict[str, str]) -> None:
    rosbags_dir = _db_root() / 'rosbags'
    rosbags_dir.mkdir(parents=True, exist_ok=True)
    _bag_meta_file().write_text(json.dumps(meta, indent=2))


def _validate_bag_name(name: str) -> str:
    if not re.fullmatch(r'[a-zA-Z0-9_\-]+', name):
        raise HTTPException(400, 'Invalid bag name')
    return name


def _clean_descriptor(text: str) -> str:
    descriptor = text.strip()
    if len(descriptor) > 200:
        raise HTTPException(400, 'Descriptor must be 200 characters or fewer')
    return descriptor


@router.get('/bags')
async def list_bags():
    rosbags = _db_root() / 'rosbags'
    descriptors = _load_bag_descriptors()
    files = _list_dir(rosbags)
    for item in files:
        item['descriptor'] = descriptors.get(item['name'], '')
    return {'files': files}


class BagDescriptorRequest(BaseModel):
    descriptor: str


@router.patch('/bags/{bag_name}')
async def update_bag_descriptor(bag_name: str, req: BagDescriptorRequest):
    bag_name = _validate_bag_name(bag_name)
    bag_path = _db_root() / 'rosbags' / bag_name
    if not bag_path.is_dir():
        raise HTTPException(404, f'Bag {bag_name!r} not found')
    descriptors = _load_bag_descriptors()
    descriptors[bag_name] = _clean_descriptor(req.descriptor)
    _save_bag_descriptors(descriptors)
    return {'ok': True, 'name': bag_name, 'descriptor': descriptors[bag_name]}


@router.delete('/bags/{bag_name}')
async def delete_bag(bag_name: str):
    bag_name = _validate_bag_name(bag_name)
    bag_path = _db_root() / 'rosbags' / bag_name
    if not bag_path.is_dir():
        raise HTTPException(404, f'Bag {bag_name!r} not found')
    shutil.rmtree(bag_path)
    descriptors = _load_bag_descriptors()
    if bag_name in descriptors:
        del descriptors[bag_name]
        _save_bag_descriptors(descriptors)
    return {'ok': True}


@router.get('/maps')
async def list_maps():
    return {'files': _list_dir(_db_root() / 'maps')}

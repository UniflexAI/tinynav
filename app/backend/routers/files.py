import os
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..state import runner

router = APIRouter(prefix='/files', tags=['files'])


def _db_root() -> Path:
    return Path(os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db'))


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


def _safe_child(root: Path, name: str) -> Path:
    if not re.match(r'^[a-zA-Z0-9_.-]+$', name):
        raise HTTPException(400, 'Invalid file name')
    root = root.resolve()
    path = (root / name).resolve()
    if path.parent != root:
        raise HTTPException(400, 'Invalid file path')
    return path


def _delete_dir(root: Path, name: str) -> dict:
    path = _safe_child(root, name)
    if not path.exists():
        raise HTTPException(404, f'{name!r} not found')
    if not path.is_dir():
        raise HTTPException(400, f'{name!r} is not a directory')
    shutil.rmtree(path)
    return {'ok': True, 'deleted': name}


@router.get('/bags')
async def list_bags():
    return {'files': _list_dir(_db_root() / 'rosbags')}


@router.get('/maps')
async def list_maps():
    return {'files': _list_dir(_db_root() / 'maps')}


@router.get('/debug-bags')
async def list_debug_bags():
    return {'files': _list_dir(_db_root() / 'debug_bags')}


@router.get('/debug-bags/{bag_name}/info')
async def debug_bag_info(bag_name: str):
    """Return ros2 bag info output for a debug bag."""
    path = _safe_child(_db_root() / 'debug_bags', bag_name)
    if not path.exists():
        raise HTTPException(404, f'{bag_name!r} not found')
    try:
        result = subprocess.run(
            ['ros2', 'bag', 'info', str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(500, f'ros2 bag info failed: {result.stderr}')
        return {'info': result.stdout}
    except subprocess.TimeoutExpired:
        raise HTTPException(500, 'ros2 bag info timed out')
    except FileNotFoundError:
        raise HTTPException(500, 'ros2 CLI not found')


@router.delete('/debug-bags/{bag_name}')
async def delete_debug_bag(bag_name: str):
    node = runner.node
    if node is not None and node.debug_recording:
        raise HTTPException(409, 'Cannot delete debug bag while recording')
    return _delete_dir(_db_root() / 'debug_bags', bag_name)


@router.delete('/bags/{bag_name}')
async def delete_bag(bag_name: str):
    node = runner.node
    if node is not None and node.state in ('realsense_bag_record', 'rosbag_build_map'):
        raise HTTPException(409, f'Cannot delete bag while in state: {node.state}')
    active_bag = node.active_bag_path if node is not None else None
    result = _delete_dir(_db_root() / 'rosbags', bag_name)
    if node is not None and active_bag is not None and Path(active_bag).name == bag_name:
        node._last_verified_bag = None
    return result


@router.delete('/maps/{map_name}')
async def delete_map(map_name: str):
    node = runner.node
    if node is not None and node.state in ('rosbag_build_map', 'navigation'):
        raise HTTPException(409, f'Cannot delete map while in state: {node.state}')
    root = _db_root()
    result = _delete_dir(root / 'maps', map_name)

    active_link = root / 'map'
    if active_link.is_symlink():
        try:
            target_name = active_link.resolve().name
            if target_name == map_name:
                active_link.unlink()
        except FileNotFoundError:
            active_link.unlink(missing_ok=True)
    return result

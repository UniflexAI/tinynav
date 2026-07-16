import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..state import runner

router = APIRouter(prefix='/files', tags=['files'])

# ── USB copy state ───────────────────────────────────────────────────────────
_usb_copy_state: dict = {'status': 'idle', 'message': '', 'detail': ''}


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


# ── USB copy ─────────────────────────────────────────────────────────────────


@router.get('/tinynav-db')
async def list_tinynav_db():
    """List top-level folders in tinynav_db."""
    root = _db_root()
    if not root.exists():
        return {'folders': []}
    folders = []
    for p in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.is_dir():
            folders.append({
                'name': p.name,
                'size': _path_size(p),
                'mtime': p.stat().st_mtime,
            })
    return {'folders': folders}


class UsbCopyRequest(BaseModel):
    folder: str  # relative path within tinynav_db, e.g. "maps/map_back"


@router.post('/copy-to-usb')
async def copy_to_usb(req: UsbCopyRequest):
    """Copy a folder from tinynav_db to USB drive, then unmount."""
    global _usb_copy_state

    if _usb_copy_state['status'] == 'running':
        raise HTTPException(409, 'A USB copy is already in progress')

    folder = req.folder.strip().strip('/')
    if not folder:
        raise HTTPException(400, 'folder is required')

    # Validate folder name (allow subpaths like maps/map_back)
    if not re.match(r'^[a-zA-Z0-9_./-]+$', folder):
        raise HTTPException(400, 'Invalid folder name')
    if '..' in folder:
        raise HTTPException(400, 'Invalid folder path')

    src = (_db_root() / folder).resolve()
    db_root = _db_root().resolve()
    if not str(src).startswith(str(db_root)):
        raise HTTPException(400, 'Invalid folder path')
    if not src.is_dir():
        raise HTTPException(404, f'Folder not found: {folder}')

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / 'scripts' / 'copy_to_usb.sh'
    if not script.exists():
        raise HTTPException(500, 'copy_to_usb.sh not found')

    _usb_copy_state = {'status': 'running', 'message': f'Copying {folder} ...', 'detail': ''}

    # Run script in background thread
    def _run():
        global _usb_copy_state
        try:
            result = subprocess.run(
                ['bash', str(script), folder],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                _usb_copy_state = {
                    'status': 'done',
                    'message': f'Copy complete: {folder}',
                    'detail': result.stdout[-500:],
                }
            else:
                _usb_copy_state = {
                    'status': 'error',
                    'message': f'Copy failed: {folder}',
                    'detail': result.stderr[-500:] or result.stdout[-500:],
                }
        except subprocess.TimeoutExpired:
            _usb_copy_state = {
                'status': 'error',
                'message': 'Copy timed out (10 min)',
                'detail': '',
            }
        except Exception as e:
            _usb_copy_state = {
                'status': 'error',
                'message': f'Copy failed: {e}',
                'detail': '',
            }

    asyncio.get_event_loop().run_in_executor(None, _run)
    return {'ok': True, 'message': 'Copy started'}


@router.get('/copy-to-usb/status')
async def copy_to_usb_status():
    """Poll USB copy progress."""
    return _usb_copy_state

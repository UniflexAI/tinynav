import os
from pathlib import Path

from fastapi import APIRouter

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


@router.get('/bags')
async def list_bags():
    return {'files': _list_dir(_db_root() / 'rosbags')}


@router.get('/maps')
async def list_maps():
    root = _db_root()
    if not root.exists():
        return {'files': []}
    entries = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / 'occupancy_grid.npy').exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        'files': [
            {
                'name': p.name,
                'size': _path_size(p),
                'mtime': p.stat().st_mtime,
                'is_dir': True,
            }
            for p in entries
        ]
    }

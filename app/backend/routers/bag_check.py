"""Read-only bag diagnostics in a cancellable process, outside the nav runtime."""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix='/files', tags=['files'])

_NAME = re.compile(r'^[a-zA-Z0-9_.-]+$')
_TOOL = Path(__file__).resolve().parents[3] / 'tool' / 'check_camera_bag.py'
_lock = asyncio.Lock()
_TIMEOUT_SECONDS = 900
_POLL_SECONDS = .5


def checked_bag(root: Path, name: str) -> Path:
    if not _NAME.fullmatch(name):
        raise HTTPException(400, 'Invalid bag name')
    root = root.resolve()
    path = (root / name).resolve()
    if path.parent != root:
        raise HTTPException(400, 'Invalid bag path')
    if not path.is_dir() or not any(path.glob('*.db3')):
        raise HTTPException(404, 'Bag not found or no SQLite bag files')
    return path


async def _stop(process):
    if process.returncode is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def run_check(path: Path, request: Request):
    if _lock.locked():
        raise HTTPException(409, 'Another bag check is running. Please wait or cancel it.')
    async with _lock:
        with tempfile.TemporaryDirectory(prefix='tinynav-bag-check-') as directory:
            result = Path(directory) / 'result.json'
            args = [
                sys.executable, str(_TOOL),
                str(path), '--skip-usb', '--json', str(result),
            ]
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            communication = asyncio.create_task(process.communicate())
            deadline = time.monotonic() + _TIMEOUT_SECONDS
            try:
                while not communication.done():
                    await asyncio.wait({communication}, timeout=_POLL_SECONDS)
                    if communication.done():
                        break
                    if await request.is_disconnected():
                        raise HTTPException(499, 'Bag check cancelled')
                    if time.monotonic() >= deadline:
                        raise HTTPException(504, 'Bag check timed out')
                stdout, stderr = await communication
                # 0 is a clean bag, 1 a bag with warnings; anything else is a failed check.
                if process.returncode not in (0, 1) or not result.is_file():
                    error = stderr.decode(errors='replace').strip()
                    raise HTTPException(422, error[-2000:] or 'Bag check could not complete')
                try:
                    report = json.loads(result.read_text())
                except (OSError, ValueError) as exc:
                    raise HTTPException(502, 'Invalid bag check result') from exc
                report.pop('usb', None)
                return {'report': report, 'text': stdout.decode(errors='replace')}
            finally:
                await _stop(process)
                if not communication.done():
                    communication.cancel()
                with suppress(asyncio.CancelledError):
                    await communication


@router.post('/bags/{bag_name}/check')
async def check_bag(bag_name: str, request: Request):
    root = Path(os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')) / 'rosbags'
    return await run_check(checked_bag(root, bag_name), request)

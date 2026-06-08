from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from fastapi import Request
from fastapi.responses import Response


BACKEND_ROLE = os.environ.get('TINYNAV_BACKEND_ROLE', 'combined').strip().lower()
if BACKEND_ROLE not in {'combined', 'manager', 'display'}:
    BACKEND_ROLE = 'combined'

MANAGER_BASE_URL = os.environ.get(
    'TINYNAV_MANAGER_BASE_URL',
    'http://169.254.10.1:8000',
).rstrip('/')

_PROXY_TIMEOUT_SEC = float(os.environ.get('TINYNAV_MANAGER_PROXY_TIMEOUT_SEC', '30'))
_HOP_BY_HOP_HEADERS = {
    'connection',
    'keep-alive',
    'proxy-authenticate',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
    'host',
    'content-length',
}


def is_display_role() -> bool:
    return BACKEND_ROLE == 'display'


def is_manager_role() -> bool:
    return BACKEND_ROLE == 'manager'


def manager_url(path: str, query: str = '') -> str:
    normalized = path if path.startswith('/') else f'/{path}'
    url = f'{MANAGER_BASE_URL}{normalized}'
    if query:
        url = f'{url}?{query}'
    return url


def _forward_headers(headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


def request_manager(
    method: str,
    path: str,
    body: bytes = b'',
    headers=None,
    query: str = '',
) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(
        manager_url(path, query),
        data=body if body else None,
        headers=_forward_headers(headers or {}),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=_PROXY_TIMEOUT_SEC) as resp:
            return resp.status, resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers.items())
    except urllib.error.URLError as e:
        body = json.dumps({'detail': f'Manager backend unavailable: {e.reason}'}).encode()
        return 502, body, {'content-type': 'application/json'}


def get_manager_json(path: str) -> dict | None:
    try:
        status, body, _ = request_manager('GET', path)
        if status < 200 or status >= 300:
            return None
        data = json.loads(body.decode('utf-8'))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def proxy_to_manager(request: Request, path: str) -> Response:
    body = await request.body()
    status, content, headers = request_manager(
        request.method,
        f'/{path}',
        body=body,
        headers=request.headers,
        query=request.url.query,
    )
    response_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
        and key.lower() not in {'content-encoding'}
    }
    return Response(content=content, status_code=status, headers=response_headers)

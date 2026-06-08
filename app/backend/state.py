"""
Global singleton — holds the NodeRunner so routers and WS handlers can
import it without circular-dependency issues.
"""
from __future__ import annotations

import os
from .node_manager import NodeRunner
from .manager_client import BACKEND_ROLE

_DEFAULT_TINYNAV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TINYNAV_DB_PATH = os.environ.get('TINYNAV_DB_PATH', os.path.join(_DEFAULT_TINYNAV_ROOT, 'tinynav_db'))

runner = NodeRunner(
    tinynav_db_path=TINYNAV_DB_PATH,
    manage_processes=BACKEND_ROLE in {'combined', 'manager'},
    telemetry_enabled=BACKEND_ROLE in {'combined', 'display'},
)

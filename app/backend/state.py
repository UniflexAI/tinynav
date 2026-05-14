"""
Global singleton — holds the NodeRunner so routers and WS handlers can
import it without circular-dependency issues.
"""
from __future__ import annotations

import os
from .node_manager import NodeRunner

_DEFAULT_TINYNAV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TINYNAV_DB_PATH = os.environ.get('TINYNAV_DB_PATH', os.path.join(_DEFAULT_TINYNAV_ROOT, 'tinynav_db'))

runner = NodeRunner(tinynav_db_path=TINYNAV_DB_PATH)

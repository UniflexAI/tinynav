"""
Global singleton — holds the NodeRunner so routers and WS handlers can
import it without circular-dependency issues.
"""
from __future__ import annotations

import os
from .node_manager import NodeRunner

TINYNAV_DB_PATH = os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')

runner = NodeRunner(tinynav_db_path=TINYNAV_DB_PATH)

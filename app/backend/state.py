"""
Global singleton — holds the NodeRunner so routers and WS handlers can
import it without circular-dependency issues.
"""
from __future__ import annotations

import os
from .node_manager import NodeRunner

TINYNAV_DB_PATH = os.environ.get('TINYNAV_DB_PATH', '/tinynav/tinynav_db')

runner = NodeRunner(tinynav_db_path=TINYNAV_DB_PATH)


class AudioState:
    """Standalone flag that forces nav-audio playback independently of
    navigation. Toggled via /nav/audio/{enable,disable}; surfaced in
    /device/status as navAudioForced so the nav-audio watcher can pick it up."""

    forced = False


audio_state = AudioState()

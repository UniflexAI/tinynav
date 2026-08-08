"""Helpers to restart the RTK bridge tmux pane used by start_app.sh."""

from __future__ import annotations

import json
import os
import subprocess


def rtk_tmux_pane() -> str:
    return os.environ.get('TINYNAV_RTK_TMUX_PANE', 'app:0.2')


def restart_rtk_bridge() -> str:
    """Kill and relaunch scripts/run_rtk.sh in the app tmux RTK pane.

    Returns the pane target used. Raises RuntimeError on failure.
    """
    pane = rtk_tmux_pane()
    cmd = [
        'tmux', 'respawn-pane', '-k', '-t', pane, '--',
        'bash', '-lc', 'cd /tinynav && /tinynav/scripts/run_rtk.sh',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except FileNotFoundError as exc:
        raise RuntimeError(f'tmux not available: {exc}') from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'tmux respawn timed out for pane {pane}') from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or 'tmux respawn failed').strip()
        raise RuntimeError(f'Failed to restart RTK on {pane}: {detail}')
    return pane


def current_active_map_name(db_root: str) -> str | None:
    link = os.path.join(db_root, 'map')
    if not (os.path.islink(link) or os.path.exists(link)):
        return None
    try:
        return os.path.basename(os.path.realpath(link))
    except OSError:
        return None


def map_rtk_enabled(map_dir: str) -> bool:
    """True when maps/<name>/nav_flow.json enables RTK (bool true or mode replace/on/...)."""
    config_path = os.path.join(map_dir, 'nav_flow.json')
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception:
        return False
    if not isinstance(config, dict):
        return False
    rtk_config = config.get('rtk', False)
    if isinstance(rtk_config, bool):
        return rtk_config
    if isinstance(rtk_config, str):
        mode = rtk_config.strip().lower()
    elif isinstance(rtk_config, dict):
        mode = str(rtk_config.get('mode', 'off')).strip().lower()
    else:
        return False
    return mode in {'replace', 'on', 'true', '1', 'yes'}


def maybe_restart_rtk_on_map_switch(
    *,
    previous_map: str | None,
    new_map: str,
    new_map_dir: str,
) -> bool:
    """Restart RTK when switching onto a different map that has RTK enabled.

    Returns True if a restart was attempted and succeeded.
    """
    if previous_map == new_map:
        return False
    if not map_rtk_enabled(new_map_dir):
        return False
    restart_rtk_bridge()
    return True

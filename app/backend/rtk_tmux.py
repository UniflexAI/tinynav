"""Helpers to restart the RTK bridge tmux pane used by start_app.sh."""

from __future__ import annotations

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

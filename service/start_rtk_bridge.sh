#!/bin/bash
# Autostart + self-heal wrapper for rtk_bridge_node.
#
# Runs the RTK bridge in a restart loop so a crash / USB replug / receiver
# hiccup auto-recovers (the container has RestartPolicy=no and this is just a
# process). Intended to be launched detached inside a tmux session so you can
# attach to see logs, e.g. from systemd:
#
#   docker exec tinynav-dev tmux new-session -d -s rtk \
#       /tinynav/service/start_rtk_bridge.sh
#   # then: docker exec -it tinynav-dev tmux attach -t rtk
#
# NTRIP credentials are read from an env file kept OUT of git
# (see rtk/.ntrip.env.example). Override paths via env vars if needed.
set -uo pipefail

source /opt/ros/humble/setup.bash 2>/dev/null || true
[ -f /tinynav/install/setup.bash ] && source /tinynav/install/setup.bash 2>/dev/null || true

# NTRIP account (TINYNAV_NTRIP_*). Kept out of git; see rtk/.ntrip.env.example.
ENV_FILE="${RTK_ENV_FILE:-/tinynav/rtk/.ntrip.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
  echo "[RTK] loaded NTRIP env from $ENV_FILE"
else
  echo "[RTK] WARN: $ENV_FILE not found; NTRIP will be unconfigured (no RTCM)."
fi

# Stable serial symlink survives ttyUSB renumbering (CH340 RTK receiver).
SERIAL="${RTK_SERIAL_PORT:-/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0}"

echo "[RTK] start_rtk_bridge at $(date -Is), serial=$SERIAL"
while true; do
  echo "[RTK] launching rtk_bridge_node at $(date -Is)"
  uv run python /tinynav/rtk/rtk_bridge_node.py --ros-args \
    -p serial_port:="$SERIAL" \
    "$@"
  code=$?
  echo "[RTK] rtk_bridge_node exited (code=$code), restarting in 2s..."
  sleep 2
done

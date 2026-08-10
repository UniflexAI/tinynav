#!/bin/bash
# Self-heal launcher for rtk_bridge_node (NTRIP + /fix,/rtk/odom,/rtk/status).
#
# Runs the RTK bridge in a restart loop so a crash / USB replug / receiver
# hiccup auto-recovers (the container has RestartPolicy=no and this is just a
# process).
#
# Autostart on the robot: the host systemd unit tinynav_start.service already
# brings up the container + web app on boot; add one line so RTK comes up the
# same way (see rtk/README.md):
#   ExecStart=-/usr/bin/docker exec -itd tinynav-dev /tinynav/scripts/run_rtk.sh
# Manual run (with logs): docker exec -it tinynav-dev /tinynav/scripts/run_rtk.sh
#
# NTRIP credentials are read from an env file kept OUT of git
# (see rtk/.ntrip.env.example). Override paths via env vars if needed.
# Source ROS BEFORE set -u: ROS setup.bash references unbound vars
# (e.g. AMENT_TRACE_SETUP_FILES) and would abort the script under nounset.
source /opt/ros/humble/setup.bash 2>/dev/null || true
[ -f /tinynav/install/setup.bash ] && source /tinynav/install/setup.bash 2>/dev/null || true

set -uo pipefail

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

# systemd starts this with `docker exec -itd`, which discards stdout, so every
# restart used to be invisible. Tee to a file that survives the container.
LOG_FILE="${RTK_LOG_FILE:-/tinynav/rtk_bridge.log}"
say() { echo "[RTK] $(date -Is) $*" | tee -a "$LOG_FILE"; }

say "start_rtk_bridge, serial=$SERIAL, log=$LOG_FILE"
while true; do
  say "launching rtk_bridge_node"
  uv run python /tinynav/rtk/rtk_bridge_node.py --ros-args \
    -p serial_port:="$SERIAL" \
    "$@" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  # 9 is the node's NMEA watchdog (NMEA_WATCHDOG_EXIT_CODE), i.e. the serial
  # link went one-way; anything else is a crash or a deliberate kill.
  if [ "$code" -eq 9 ]; then
    say "*** NMEA WATCHDOG FIRED -- serial went silent, relaunching in 2s ***"
  else
    say "rtk_bridge_node exited (code=$code), restarting in 2s..."
  fi
  sleep 2
done

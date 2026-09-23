#!/bin/bash
# The bridge's raw pty must have exactly one reader: two readers split the lines.
HERE=$(cd "$(dirname "$0")" && pwd)
while true; do
  # A reader orphaned by the previous round would keep stealing half the lines.
  docker exec tinynav-dev pkill -f "^cat /tmp/rtk_nmea" 2>/dev/null
  docker exec tinynav-dev cat /tmp/rtk_nmea 2>/dev/null | python3 -u "$HERE/rtk_diag_logger.py"
  sleep 2
done

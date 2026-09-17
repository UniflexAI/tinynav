#!/bin/bash
# One A/B capture. Usage:  rtk_ab_run.sh with_looper     (Ctrl-C to finish)
#
# Everything the diagnosis needs, timestamped against one wall clock: per-band
# C/N0 (the bridge's CSV), NTRIP reachability split into "local hop" vs "MiFi
# backhaul", motion/IMU for the vibration hypothesis, and the kernel's USB view.
set -u
LABEL="${1:?usage: rtk_ab_run.sh <label>, e.g. with_looper / no_looper}"
STAMP=$(date +%m%d_%H%M%S)
# Under the repo, because that is the only host path the container can see
# and ros2 bag record runs inside it.
HOST_ROOT=/home/dm/workspace/tinynav
OUT="$HOST_ROOT/rtk_ab/${LABEL}_${STAMP}"
CONT_OUT="/tinynav/rtk_ab/${LABEL}_${STAMP}"
mkdir -p "$OUT"

BRIDGE_LOG=$HOST_ROOT/rtk_bridge.log
SIGNAL_CSV=$HOST_ROOT/rtk_signal.csv
GW=$(ip route | awk '/^default/{print $3; exit}')
CASTER=$(grep -oE '^export TINYNAV_NTRIP_HOST=.*' $HOST_ROOT/rtk/.ntrip.env 2>/dev/null | cut -d= -f2- | tr -d "\"'")
: "${CASTER:=120.253.239.161}"
CPORT=$(grep -oE '^export TINYNAV_NTRIP_PORT=.*' $HOST_ROOT/rtk/.ntrip.env 2>/dev/null | cut -d= -f2- | tr -d "\"'")
: "${CPORT:=8002}"

# Remember where the shared, append-only logs stood, so the slice at the end is
# exactly this run and not everything since boot.
BRIDGE_OFF=$(stat -c %s "$BRIDGE_LOG" 2>/dev/null || echo 0)
SIGNAL_OFF=$(stat -c %s "$SIGNAL_CSV" 2>/dev/null || echo 0)
echo "$BRIDGE_OFF $SIGNAL_OFF" > "$OUT/offsets.txt"

{
  echo "label=$LABEL  start=$(date -Is)  gw=$GW"
  echo "--- uptime ---";      uptime
  echo "--- lsusb -t ---";    lsusb -t
  echo "--- lsusb ---";       lsusb
  echo "--- interfaces ---";  ip -br addr
  echo "--- wifi ---";        iw dev wlP1p1s0 link 2>/dev/null
  echo "--- git ---";         git -C $HOST_ROOT rev-parse --short HEAD
} > "$OUT/env_before.txt" 2>&1

# Reachability, 1 Hz. The gateway is the MiFi itself; if it answers while the
# caster does not, the outage is the MiFi's cellular backhaul, not our radio.
(
  echo "wall,gw_ms,caster_tcp,rssi_dbm"
  while true; do
    g=$(ping -c1 -W1 "$GW" 2>/dev/null | grep -oP 'time=\K[0-9.]+' || echo NA)
    # The caster drops ICMP, so probe what we actually depend on: a TCP
    # handshake to the NTRIP port. 1 = reachable, 0 = not.
    if timeout 2 bash -c "exec 3<>/dev/tcp/$CASTER/$CPORT" 2>/dev/null; then c=1; else c=0; fi
    r=$(iw dev wlP1p1s0 link 2>/dev/null | grep -oP 'signal: \K-?[0-9]+' || echo NA)
    echo "$(date +%s.%N),$g,$c,$r"
    sleep 1
  done
) > "$OUT/net.csv" &
NETPID=$!

# Motion + power. Small topics only: /lf/lowstate carries the IMU, which is how
# we test "vibration rattles the antenna connector".
docker exec tinynav-dev bash -lc \
  "source /opt/ros/humble/setup.bash; cd $CONT_OUT && ros2 bag record -o ctx \
   /cmd_vel /battery /nav/active /fix /rtk/odom /rtk/map_pose /lf/lowstate \
   /mapping/current_pose_in_map /rtk/io_status" > "$OUT/bag.log" 2>&1 &

cleanup() {
  echo
  echo "[rtk_ab] stopping..."
  kill "$NETPID" 2>/dev/null
  docker exec tinynav-dev pkill -INT -f "ros2 bag record -o ctx" 2>/dev/null
  sleep 3
  tail -c "+$((BRIDGE_OFF + 1))" "$BRIDGE_LOG" > "$OUT/rtk_bridge.log" 2>/dev/null
  if [ "$SIGNAL_OFF" -gt 0 ]; then
    head -1 "$SIGNAL_CSV" > "$OUT/rtk_signal.csv"
    tail -c "+$((SIGNAL_OFF + 1))" "$SIGNAL_CSV" >> "$OUT/rtk_signal.csv"
  else
    cp "$SIGNAL_CSV" "$OUT/rtk_signal.csv" 2>/dev/null
  fi
  echo 0731 | sudo -S dmesg -T > "$OUT/dmesg.txt" 2>/dev/null
  { echo "end=$(date -Is)"; echo "--- lsusb -t ---"; lsusb -t; } > "$OUT/env_after.txt" 2>&1
  # The bag is written by root inside the container; hand it back or the
  # host user cannot copy or delete the run.
  docker exec tinynav-dev chown -R "$(id -u):$(id -g)" "$CONT_OUT" 2>/dev/null
  echo "[rtk_ab] done -> $OUT"
  ls -la "$OUT"
  exit 0
}
trap cleanup INT TERM

echo "[rtk_ab] recording '$LABEL' -> $OUT"
echo "[rtk_ab] gateway=$GW caster=$CASTER:$CPORT"
echo "[rtk_ab] press Ctrl-C when the run is over"
while true; do sleep 5; done

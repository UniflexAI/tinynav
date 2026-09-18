#!/usr/bin/env bash
# Fully stop every simulation process in this container -- without restarting
# it. run_simulator.sh runs this first (relaunch semantics); it is also the
# manual "stop the last sim" tool. Exits non-zero if anything survived.
#
# Safe against the pkill self-match trap: the patterns never appear in this
# script's own argv (it runs as `bash .../kill_sim.sh`).
PATTERNS=(
  'ign gazebo'
  'parameter_bridge'
  'robot_state_publisher'
  'robot_controller_gazebo'
  'cmd_vel_pub'
  'go2_controller'
  'gt_twist_pub'
  'cmd_vel_servo'
  'controller_manager spawner'
  'perception_node'
  'camera_info_publisher'
  'scene_runner'
  'sim_gt_reloc'
  'keyboard_teleop'
  'ros_gz_sim create'
  'ros2 topic pub'
  'ros2 service call'
  'record_session'
  'verify_vio'
)

for p in "${PATTERNS[@]}"; do pkill -TERM -f "$p" 2>/dev/null; done
sleep 2
for p in "${PATTERNS[@]}"; do pkill -KILL -f "$p" 2>/dev/null; done
tmux kill-session -t tinynav_sim 2>/dev/null

# FastDDS shared-memory segments whose owner process is gone: leftover
# segments slow the next DDS init and can alias stale topics.
for f in /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; do
  [ -e "$f" ] || continue
  if ! grep -qs "$f" /proc/*/maps 2>/dev/null; then
    rm -f "$f"
  fi
done

sleep 1
LEFT=0
for p in "${PATTERNS[@]}"; do
  if pgrep -f "$p" >/dev/null 2>&1; then
    echo "still alive: $p"
    LEFT=1
  fi
done
if tmux has-session -t tinynav_sim 2>/dev/null; then
  echo "tmux session tinynav_sim still alive"
  LEFT=1
fi
if [ "$LEFT" = 0 ]; then
  echo "sim fully stopped (no gz, no nodes, no tmux session)"
fi
exit "$LEFT"

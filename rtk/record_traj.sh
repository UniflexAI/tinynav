#!/bin/bash
set -eo pipefail

# Record RTK topics for offline trajectory verification.
#
# Usage:
#   bash /tinynav/rtk/record_traj.sh [output_name]
#
# Env overrides:
#   RTK_BAG_DIR   parent directory for the bag (default: /tinynav/output)
#
# Start the RTK bridge first (scripts/run_rtk.sh), wait until /rtk/status
# shows gga_quality 4 (RTK fixed), then run this while you walk the path.

# ROS 2 setup.bash references unbound shell variables, so it MUST be sourced
# before enabling `set -u` (nounset); otherwise the whole script aborts on the
# source line with no output.
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
set -u

name="${1:-rtk_traj_$(date +%m%d_%H%M%S)}"
bag_dir="${RTK_BAG_DIR:-/tinynav/output}"
out="$bag_dir/$name"

topics=(
  /fix
  /rtk/odom
  /rtk/path
  /rtk/status
  /rtk/io_status
  /rtk/receiver_status
  /vel
  /time_reference
)

mkdir -p "$bag_dir"

echo "[RTK] output bag : $out"
echo "[RTK] topics     : ${topics[*]}"
echo "[RTK] Make sure the bridge is publishing and RTK is fixed before you move."
echo "[RTK] Ctrl-C to stop recording."
echo

exec ros2 bag record -o "$out" "${topics[@]}"

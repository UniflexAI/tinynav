#!/bin/bash
set -eo pipefail

source /opt/ros/humble/setup.bash
if [[ -f /3rdparty/message_filters_ws/install/local_setup.bash ]]; then
  source /3rdparty/message_filters_ws/install/local_setup.bash
fi

if ! command -v ros2 >/dev/null; then
  echo 'ros2 not found after sourcing ROS; aborting.' >&2
  exit 1
fi

while ! ros2 node list | grep -qx '/insight_full'; do
  echo '/insight_full not found, retrying in 2s...'
  sleep 2
done

echo '/insight_full detected, starting TinyNav app...'
exec /tinynav/scripts/start_app.sh

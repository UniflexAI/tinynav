#!/usr/bin/env bash
set -eo pipefail

APP_ROOT="/userdata/junlinp/tinynav"
DB_PATH_DEFAULT="/userdata/junlinp/tinynav_db"
VENV_PATH_DEFAULT="/userdata/junlinp/venv"

# Ensure ROS 2 Python packages (rclpy, msgs) are on PYTHONPATH.
if [ -f /opt/ros/humble/setup.bash ]; then
  # setup.bash may reference unset vars; avoid nounset during source.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
else
  set -u
fi

cd "$APP_ROOT"

export VENV_PATH="${VENV_PATH:-$VENV_PATH_DEFAULT}"
if [ -x "${VENV_PATH}/bin/python3" ]; then
  export PATH="${VENV_PATH}/bin:${PATH}"
  export VIRTUAL_ENV="${VENV_PATH}"
fi

export PYTHONPATH="$APP_ROOT:${PYTHONPATH:-}"
export TINYNAV_DB_PATH="${TINYNAV_DB_PATH:-$DB_PATH_DEFAULT}"
export TINYNAV_BACKEND_ROLE="${TINYNAV_BACKEND_ROLE:-manager}"
export LD_LIBRARY_PATH="/userdata/opencv-release/lib:/userdata/hobot/opt/hobot/deps:${LD_LIBRARY_PATH:-}"

exec python3 -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

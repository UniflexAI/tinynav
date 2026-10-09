# The app image plus robot_localization, so the Go2 EKF runs beside the app without touching its container.
# diagnostic-updater must be named: ekf_node fails on a missing libdiagnostic_updater.so otherwise.
FROM uniflexai/tinynav:latest
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-robot-localization ros-humble-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*

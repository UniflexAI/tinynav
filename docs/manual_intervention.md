# Manual intervention

Open Operate and select **Take control**. TinyNav stops only its automatic
velocity controller. Sensors, localization, local planning, and map navigation
continue running and updating the display. Manual movement can reach a POI,
advance the queue, and complete the task normally. Taking control also clears
an existing task pause so that POI tracking can continue.

Hold W/S for forward/backward, A/D for left/right translation, and Left/Right
arrow keys for rotation. The joysticks remain available. Linear speed is limited
to 0.5 m/s (including diagonals) and rotation to 1 rad/s. Releasing keys,
leaving the control page, or losing browser focus sends zero velocity. The
backend independently publishes zero after 300 ms without renewed input.
Stopping manual input keeps manual ownership; use **Exit manual** to hand
control back to autonomous navigation.

Select **Exit manual** to restore the automatic controller. An unfinished task
continues immediately without a separate Resume action. The new controller
starts with empty velocity/path caches and waits for a fresh planner path.
If manual movement completed the task, exiting does not restart it. With no
task, exiting manual control returns to idle. **Cancel** explicitly cancels a
task. The queue is held in the running map process; backend/container restart
recovery is outside this feature.

Manual mode blocks starting another task, changing maps, recording, or
rebuilding maps. Navigation must be enabled before sending POIs. Legacy ROS
navigation commands cannot launch a second autonomous stack or bypass manual
ownership. External ROS publishers are outside this process manager's ownership.

## API and verification

- `POST /nav/manual/enter`
- `POST /nav/manual/exit`
- `GET /device/status`: `manualIntervention`, `navPaused`, `navActive`
- `WS /ws/teleop`: `linear_x`, `linear_y`, `angular_z`

Run in the development container with the ROS environment loaded:

```bash
python3 -m unittest tests/test_manual_intervention.py
cd app/frontend
flutter analyze --no-pub --no-fatal-infos
```

Build the web client with `bash scripts/build_frontend.sh`. This versions the
compiled JavaScript URL so a page reload picks up the current application.

## Home Gazebo

Use `TINYNAV_SENSOR_MODE=looper`, `TINYNAV_DISABLE_CHASSIS=1`, and
`ROBOT_TYPE=lekiwi`. The simulation adapter and planar model configuration are
in tinynav-sim's `tools/manual_intervention/`. They use Gazebo ground truth for
odometry and camera depth; this does not simulate VIO failure or recovery.

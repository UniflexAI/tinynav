# Manual intervention

Open Operate and select **Take control**. TinyNav pauses the task and stops its
local planner and automatic velocity controller. Sensors and map localization
continue running. The map process retains the POI queue and current POI index;
its navigation timer cannot advance the task while paused.

Hold W/S for forward/backward, A/D for left/right translation, and Left/Right
arrow keys for rotation. The joysticks remain available. Linear speed is limited
to 0.5 m/s (including diagonals) and rotation to 1 rad/s. Releasing keys,
leaving the control page, or losing browser focus sends zero velocity. The
backend independently publishes zero after 300 ms without renewed input.

Select **Exit manual** to restore the planner and automatic controller. An
existing task stays paused until **Resume** is selected. The map planner
invalidates its cached route and plans from the current pose; the controller
clears cached velocities and waits for a fresh path. With no task, exiting
manual control returns to idle. **Cancel** explicitly cancels a retained task.
The task is retained in the running map process, not a checkpoint for recovery
across backend/container restarts.

Manual mode blocks starting navigation, changing maps, recording, or rebuilding
maps. Navigation must be enabled before sending POIs. Legacy ROS navigation
commands cannot launch a second autonomous stack or bypass manual ownership.
External ROS publishers are outside this process manager's ownership.

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

# Stair mode with a floor count

This branch starts from `xiaole/stair-mode`. It uses the existing
`stair_node -> /control/target_pose -> planning_node -> cmd_vel_control` pipeline.

In the App, open **Stairs**, select the floor count and turn side, then choose
**Down**. The start dialog defaults to one floor when the saved setting is unlimited.
Each run can override the saved settings. The existing settings page controls
camera height and landings per floor.

The initial trial assumes a start on a full-floor landing and two flights per
floor. The intermediate landing counts as one flight. One floor completes after
two landings with the requested height change and a sustained return to level
attitude. The App retains the arrival result after leaving stair mode.
Other layouts can set landings per floor to 1 through 4. A count of zero retains
the previous unlimited mode.

The start API accepts:

```json
{"direction": "down", "turn": "auto", "floors": 1, "landings_per_floor": 2}
```

Send it to `POST /nav/stair/start`. Older clients that omit the count use the
saved settings. `POST /nav/pause`, `/nav/resume`, and `/nav/stair/stop` remain the
pause, continue, and end controls.

Depth that has stopped updating for 0.75 seconds clears the target, even if the
odometry stream remains alive. Stop statuses zero the velocity controller
immediately. A landing search with no translation or turning for 6 seconds
discards its stuck goal and temporarily searches reachable, observed ground.
All resulting motion still passes through the existing planner.

## Verification on Spark

The focused tests cover counts, pause/resume, depth loss, odometry jumps,
goal recovery, API compatibility, and retained arrival results:

```bash
python3 -m pytest -q tests/test_stair_mode.py
```

The Go2 integration check requires the MuJoCo runtime and course builder from
the independent simulation worktree:

```bash
PYTHONPATH=$PWD:/path/to/tinynav-stairs:/path/to/mujoco/dependencies \
python3 tool/stair_mode_sim.py \
  --sim-root /path/to/tinynav-sim \
  --course-script /path/to/tinynav-stairs/scripts/simulate_stair_descent.py
```

Run inside a localhost-only ROS domain with the Go2 ONNX runtime and EGL renderer.
The check uses one calibrated front camera, simulated odometry, the original
target generator and planner, and the Go2 gait. It asserts two completed flights,
arrival, zero velocity, a stable 1.5-second hold, no wall contact, and positive
shaft clearance.

The App Web build and this DOWN integration check pass. Upstairs physical
operation and real L-corner recovery still need field validation. The older
backend suite has four failures also present on the base commit, covering map
image dimensions, yaw conversion, and two map-build API fixtures.

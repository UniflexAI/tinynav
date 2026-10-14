# Timed planner to cmd_vel integration

Branch: `xiaole/continuous-planner-websim`.

## Software chain

`map_node (route target) -> planning_node -> /planning/trajectory_path ->
cmd_vel_control -> /cmd_vel -> existing Unitree transport -> SportClient.Move`.

WebSim now launches the real `cmd_vel_control.py`, rather than its separate
simulator controller. GO2 SDK transport is not launched on spark.

### Trajectory contract

Continuous paths publish every 0.1-second pose with a distinct future timestamp.
The first pose timestamp is the source depth/odometry time. Frame is `world`;
position is control-center position and orientation retains the existing camera
axis convention. Controller derives velocity from adjacent poses and timestamp
difference, without depending on a planner environment variable. Old DWA paths
with identical timestamps retain the 1.0-second legacy interpretation.

Timed paths older than 0.25 seconds, invalid intervals, invalid poses or unexpected
frames stop control. Timed-path silence for 0.35 seconds stops, rather than emitting
sub-deadzone slow commands. Odometry silence and legacy-path expiry also stop.
Navigation deactivation sends one stop then releases /cmd_vel for manual control.
Pause and resume discard prior commands; resume requires a new path. A stationary
path commands an immediate stop, without leaving an acceleration-limited tail.

Continuous trajectories already include rotate-first behavior, so legacy
heading-error overrides are not reapplied. Forward acceleration limiting remains
in the controller. Repeated unpaused status messages no longer reset acceleration.

### Physical deadzones

`RobotConfig.reverse_speed` defaults to 0.3 m/s. Continuous recovery predicts
that same velocity, validates a 0.24 m body sweep (0.8 s), and limits each attempt
to 0.6 s with refreshed validation. The predicted path holds position after 0.8 s.
At most three attempts per target are allowed. Unknown or blocked rear space
rejects recovery. A timed path requesting a different reverse speed is rejected,
not silently amplified. Legacy DWA reverse behavior remains 0.3 m/s.

Forward and yaw deadzones are applied in the planner rollout using RobotConfig.
Near a target, aligned forward motion can use the minimum effective speed rather
than an unexecutable small speed. A* uses the actual target coordinate within the
terminal cell, avoiding systematic grid-center endpoint error.

Unitree bridge adds a 0.5-second command-loss watchdog that calls StopMove once
if a moving command stream disappears. Logic is tested with a fake SDK client;
actual hardware execution remains to be verified.

## Route guidance

The existing map_node selector is extracted unchanged into
`tinynav/core/navigation_guidance.py`, shared by WebSim. Its current lookahead
is approximately 2.5 m accumulated along the route, not a straight-line target.
WebSim accepts a JSON `guide_path` containing dense `[x,y,z]` route samples.
Without it, existing direct-target pressure tests remain available. The simulator
publishes at its own tick rate; this does not reproduce map localization or all
map_node timing. No new global route is inferred from unknown space.

## Validation

- 13 controller/guidance/bridge-logic tests plus 9 existing prototype tests pass.
- Shared-controller L-turn and S-bend repeated twice: all reached, no collision.
- Actual ROS WebSim fault smoke: pause stop, resume, sensor-loss stop, sensor
  recovery and navigation deactivation stop all passed.
- Map-guided L turn reached in 19.7 s, target error 0.078 m, no collision.
- Final endpoint/deadzone regression: L turn reached in 21.2 s and S bend
  reached in 14.1 s, no collisions or blocked samples.
- Raw comparison results and fault samples are saved with this document.

The existing Python/C++ raycast comparison also fails on unchanged main in this
environment; no raycast backend change is included here.

## Enable in the robot application

Inside the robot deployment container, from `/tinynav`:

```bash
TINYNAV_PLANNER=continuous bash scripts/start_app.sh
```

The script now explicitly carries the selected planner into the backend process,
whose child nodes inherit it. Existing application/transport nodes must first be
stopped to avoid duplicate publishers. Normal navigation activation/pause remains
owned by the application. `TINYNAV_PLANNER=dwa` selects the previous planner.
This command is documented; it has NOT been executed on a robot in this task.

## Remaining hardware and performance work

This completes the software control interface and shared-controller simulation;
it is not evidence of GO2 physical tracking performance. No GO2 movement was
commanded. Validate transport/topic interoperability, real deadzones, tracking,
physical stopping distance, sensor delays and planner latency on the target robot.
B2 L-turn behavior and Nano performance are not solved by this interface change.
Map dimensions and optimization length have not yet been scaled by length/speed.
Acceleration/jerk-constrained trajectory optimization is not implemented.

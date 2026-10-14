# Timed planner to cmd_vel integration

Branch: `xiaole/continuous-planner-websim`.

## Software chain

`map_node (route target) -> planning_node -> /planning/trajectory_path ->
cmd_vel_control -> /cmd_vel -> existing Unitree transport -> SportClient.Move`.

WebSim now launches the real `cmd_vel_control.py`, rather than its separate
simulator controller. GO2 SDK transport is not launched on spark.

### Trajectory contract

Continuous paths publish every 0.1-second pose with a distinct future timestamp.
Path and pose timestamps use the ROS publication clock; sensor timestamps may
use a separate hardware epoch. Non-increasing sensor timestamps do not refresh a
trajectory. The first two sensor frames build occupancy before motion is enabled. Frame is `world`;
position is control-center position and orientation retains the existing camera
axis convention. Controller derives velocity from adjacent poses and timestamp
difference, without depending on a planner environment variable. Old DWA paths
with identical timestamps retain the 1.0-second legacy interpretation.

Timed paths older than 0.25 seconds, invalid intervals, invalid poses or unexpected
frames stop control. Timed-path silence stops at a period-adaptive deadline: at least 0.35 seconds,
scaled by observed receive period, capped at 0.8 seconds and the first planned
stationary segment (minus publication age). Receive-period learning survives
stops. Commands advance through the timed sequence, rather than indefinitely
holding its first segment. For slower updates, forward playback rate is reduced
using a 0.2-second reference period, without dropping below the forward deadzone.
Reverse playback remains at its validated 0.3 m/s. No extrapolation is allowed
beyond the validated horizon. Odometry silence and legacy-path expiry also stop.
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

## Low-rate controller regression

33 targeted tests pass (23 control, 7 continuous-planner, 3 mapping/timing).
With WebSim sensor_period_s=0.4 (approximately 0.5 s between frames on its
8 Hz timer), final L turn reached in 52.6 s with 0.091 m target error and no
collision. Raw samples: `docs/pnc-rate-adaptive-final.json`. Initial low-rate
trials exposed unsafe startup mapping and endpoint deadzone roundoff; the final
version waits for three sensor frames before moving and tolerates 1e-6 velocity
roundoff at the minimum-speed boundary. These are simulation regressions, not
physical GO2 validation.

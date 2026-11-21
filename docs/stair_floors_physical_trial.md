# Physical stair trial and recording handoff

For this App/stairs-mode update, including the existing-device upgrade steps,
see [App upgrade and use (Chinese)](stair_app_upgrade_zh.md). Updating Python
sources and restarting backend/navigation processes is required in addition to
rebuilding the ordinary Flutter frontend.

Updated: 2026-11-21. Read alongside
[the complete x64 deployment guide](stair_floors_ubuntu_x64_deployment.md).

## What is and is not deployed

Deploy this branch's App, Looper bridge, stair target generator, local planner
and cmd_vel tracking. The simulation Go2 ONNX policy is NOT the physical gait.
The physical Unitree path uses SportClient/ClassicWalk; first establish that the
actual robot/controller can negotiate the intended stairs. UP/DOWN 9 success in
MuJoCo does not establish this. Real depth/VIO, DDS transport, different friction,
foot placement and camera mount remain field variables.

The simulation's shaft/contact checker knows world geometry and halts the test.
It is not present as an independent shaft detector on the robot. This branch's
Unitree bridge also has no independent command-loss timer watchdog. Verify the
robot/controller's actual loss-of-command stop and physical stop before stairs.

## Install and connect without starting a stair task

1. Use a separate clean `xiaole/stair-floors` checkout. Record full commit and
   amd64 image digest. Follow deployment sections 3-6 for Docker/GPU, ROS overlays,
   native dependencies, model assets, frontend build and the focused tests in the linked upgrade guide.
2. Initially set `ROBOT_TYPE=go2`, `TINYNAV_UNITREE_ENABLED=0`. Use the actual
   sensor ROS domain/network, not the simulation's localhost-only domain 79.
   Source ROS/driver overlays in every launch shell. No simulation process should
   share a physical command network.
3. Start the existing vendor Looper driver (not included here) and confirm
   `/insight_full` discovery before App startup. Check the exact topic/type/stamp,
   depth units and camera-optical/world-axis contract in deployment section 7.
   Depth, left image and `/camera/camera/vio_image` must have identical stamps;
   `/camera/camera/vio_100hz` must remain fresh and continuous. Check QoS using
   `ros2 topic info -v`; a plausible image does not prove synchronized geometry.
4. Verify actual camera height, mount and robot body/control-center dimensions.
   `STAIR_CAMERA_HEIGHT=0.4669` is the tested reference only. Saved App settings
   override that variable; planner camera offsets/body dimensions are separate
   in `robot_specs.py`. Compare stationary floor height, wall direction and VIO
   movement on flat ground before approaching stairs.
5. Start App with section 8, inspect child logs and `/slam/depth`,
   `/slam/odometry_visual`, `/slam/odometry`, camera_info and targets/paths.
   Keep Unitree disabled while checking outputs. Settings use relative floors:
   one floor normally means two flights, starting on a full-floor platform.
6. Select one physical command owner: existing proven actuator bridge OR
   App-managed Unitree. For the latter, set `TINYNAV_UNITREE_ENABLED=1` and the
   actual `UNITREE_NETWORK_INTERFACE`; recreate only this deployment container
   (restart does not reload env). Section 10 explains NIC/domain/type checks.
   Verify zero/stop, forward/yaw signs, pause and command-loss behavior on flat
   ground. No automatic stair start is included in deployment.

## First supervised stair task

Use a finite task from a full-floor platform, with a person holding the robot's
physical stop and a clear recovery area. Start with one flight only through a
separately configured/tested counting convention, or the normal one-floor task
(two flights); do not interpret `floors:1` as one flight. Avoid a first physical
nine-floor run. Retain freshness, planner and stop checks; do not lower count
thresholds or remove checks to accommodate bad sensor data.

Start recording BEFORE task start, including a stationary segment. In a shell
on the backend host/container (replace the backend address when remote):

```bash
curl --fail http://127.0.0.1:8000/field-record/info
curl --fail -X POST http://127.0.0.1:8000/field-record/start
curl --fail -X PUT http://127.0.0.1:8000/nav/stair/settings \
  -H 'Content-Type: application/json' \
  -d '{"camera_height":0.4669,"floors":1,"landings_per_floor":2}'
# Only after checking the real height/settings and being ready to move:
curl --fail -X POST http://127.0.0.1:8000/nav/stair/start \
  -H 'Content-Type: application/json' \
  -d '{"direction":"down","turn":"auto","floors":1,"landings_per_floor":2}'
```

Use `direction:"up"` for a separate upward trial. `turn` is relative to the
incoming flight. Pause: `POST /nav/pause`; resume: `POST /nav/resume`; leave stair
mode: `POST /nav/stair/stop`. App pause/stop needs a working software/control
path and is not a replacement for the physical stop.

At the last count, `finishing` can briefly continue forward onto measured flat
support; only then does `arrived` latch. Missing support produces `no_seed` and
zero output, not success. Count completion alone is no longer the arrival signal.
If VIO/depth is stale, the existing freshness stop still takes precedence.

After stopping the robot, keep a short stationary recording segment, then:

```bash
curl --fail -X POST http://127.0.0.1:8000/field-record/stop
curl --fail http://127.0.0.1:8000/field-record/info
```

The bag is under `/tinynav_data/debug/field_*` (host data mount), with `info.json`.
Inspect `ros2 bag info BAG_DIRECTORY`: check duration, message counts and actual
input/output topics. The API being marked recording does not guarantee a sensor
was publishing. Keep the entire directory, including metadata.yaml and all db3
or mcap segments. Do not send only a screen recording or a selected database file.

## Recording needed for analysis

The App field recorder includes raw left/depth/calibration, synchronized/raw VIO,
IMU/status when available, tf_static, stair cmd/status/path, target, trajectory,
obstacle mask, cmd_vel, navigation active/paused and optional leg/EKF topics.
See `BackendNode.FIELD_RECORD_TOPICS` for the exact versioned list. It does not
currently include bridge `/slam/*` outputs or `/planning/height_map`.

For the first field diagnosis, optionally run this additional recorder in a
second sourced ROS shell, in the same physical ROS domain, before motion:

```bash
mkdir -p /tinynav_data/debug
ros2 bag record --output /tinynav_data/debug/derived_$(date +%Y%m%d_%H%M%S) \
  /slam/depth /slam/odometry_visual /slam/odometry \
  /camera/camera/infra2/camera_info /planning/height_map /tf /tf_static
```

Stop this recorder with Ctrl-C and wait for clean closure. Confirm publishers
and QoS; do not assume optional topics exist. Check disk space before recording
raw and derived depth simultaneously. If the extra depth volume is too large,
the App's raw synchronized inputs are the priority for replay.

Return these together:

- Complete bag directory/directories, `info.json`, and `ros2 bag info` output.
- Full source SHA, `git status --short`, any local patch, image digest, robot and
  controller/firmware version, actual gait/control mode. Exclude credentials.
- Actual task JSON and stair settings, starting on full/half landing, real
  flight count, stair width/rise/run/platform dimensions and turn direction.
- Camera intrinsics/extrinsics/height, robot footprint and planner offsets;
  relevant env (robot type, domain, NIC, RMW), sensor/node/topic QoS inventory.
- `/tinynav_data/logs` for Looper/planner/stair/control/Unitree/backend, including
  a few seconds before and after the event. Keep failure evidence unchanged.
- A side/front video showing feet, stair edge and landing, plus BEV/App video
  if possible. State event time in bag time or seconds after task start, observed
  behavior, whether manual intervention occurred, and expected behavior.

This permits separating sensor timing/axes/VIO drift, target selection, path
planning, tracking and gait slip. A ROS cmd_vel trace alone cannot prove the
SportClient received it or the feet executed it; include controller-side feedback
or logs from the actual actuator bridge when available. Replay bags only with
physical actuation disconnected/disabled in an isolated domain.

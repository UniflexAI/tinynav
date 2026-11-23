# Physical stair trial and recording handoff

For this App/stairs-mode update, including the existing-device upgrade steps,
see [App upgrade and use (Chinese)](stair_app_upgrade_zh.md). Updating Python
sources and restarting backend/navigation processes is required in addition to
rebuilding the ordinary Flutter frontend.

Updated: 2026-11-23. Read alongside
[the complete x64 deployment guide](stair_floors_ubuntu_x64_deployment.md).

## What is and is not deployed

Deploy this branch's App, Looper bridge, stair target generator, local planner
and cmd_vel tracking. The simulation Go2 ONNX policy is NOT the physical gait.
The physical Unitree path uses SportClient/ClassicWalk; first establish that the
actual robot/controller can negotiate the intended stairs. UP/DOWN 9 success in
MuJoCo does not establish this. Real depth/VIO, DDS transport, different friction,
foot placement and camera mount remain field variables.

The simulation's shaft/contact checker knows world geometry and halts the test.
It is not present as an independent shaft detector on the robot. The VIO guard now gates both cmd_vel and the Unitree bridge, including a 1 s
guard-heartbeat timeout. It does not measure physical stopping distance or
replace a general SDK/DDS command-loss watchdog. Verify the actual controller's
stop behavior. See [VIO stop and automatic recovery](stair_vio_safety.md).

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
5. Start App with section 8 (it starts the persistent VIO guard automatically), inspect child logs and `/slam/depth`,
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

In the App, open **Operate**. The fixed bar above the camera has **Record bag**.
Start it BEFORE the task and include a short stationary segment. The button turns
red (**Stop & save**) and displays `REC mm:ss`. Leaving Operate does not stop the
recorder. Configure the actual camera height and a small task in **Stairs**, then
start the upward or downward trial. Recording itself does not start motion.

For shell/API diagnosis, the same recorder is available at
`POST /field-record/start`, `POST /field-record/stop` and `GET /field-record/info`
on the backend (normally port 8000). The App needs both this backend revision and
the rebuilt frontend; rebuilding Flutter alone does not update recorder topics.

Use `direction:"up"` for a separate upward trial. `turn` is relative to the
incoming flight. Pause: `POST /nav/pause`; resume: `POST /nav/resume`; leave stair
mode: `POST /nav/stair/stop`. App pause/stop needs a working software/control
path and is not a replacement for the physical stop.

At the last count, `finishing` can briefly continue forward onto measured flat
support; only then does `arrived` latch. Missing support produces `no_seed` and
zero output, not success. Count completion alone is no longer the arrival signal.
If VIO/depth is stale, the existing freshness stop still takes precedence.

After stopping the robot, keep a short stationary segment, then press
**Stop & save**. Wait for **Saved · N messages**. The backend closes rosbag using
SIGINT and verifies metadata and nonzero messages with `ros2 bag info`; an empty
or incomplete bag produces an error instead of a success label. **Check inputs**
means the bag is readable but one or more basic input topics had no messages.
Use the folder icon to inspect missing topics and **Copy path**.

Files stay on the robot/backend host, normally `/tinynav_data/debug/field_*`
(the configured DB mount), rather than downloading to the phone/browser.
Return the complete folder: `metadata.yaml`, every db3/mcap segment, and
`info.json`. The latter records source revision/dirty files, start settings,
latest stair state, requested topics, actual message counts and verification.
For an independent inspection, use `ros2 bag info BAG_DIRECTORY`.

## Recording needed for analysis

The single Operate recorder requests 38 topics: raw left/right/depth/calibration,
RGB, synchronized/raw VIO, IMU/status, `/tf` and `/tf_static`, stair command/status/path,
target/trajectory/obstacle mask, `/cmd_vel`, navigation flags, optional leg/EKF,
and bridge `/slam/depth`, `/slam/odometry_visual`, `/slam/odometry` plus
`/planning/height_map` and `/nav/vio_safety`. No second derived-data recorder is needed.
`BackendNode.FIELD_RECORD_TOPICS` is the exact versioned list.

`/sportmodestate` is requested as optional native Unitree feedback; it only appears
if a compatible ROS publisher/type is actually available. It does not prove the
SDK subscriber received or executed `/cmd_vel`. Confirm actual publishers and QoS.
The App warns about absent depth, visual odometry, odometry, infrared2 CameraInfo
and cmd_vel messages; passing that check does not validate their calibration or
quality. Check available disk space before recording raw and derived depth.

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

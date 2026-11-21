# Stair-floors deployment handoff for Ubuntu x64

Updated: 2026-11-21. Intended reader: the GPT/Codex agent running on the destination computer.

## 1. Read this first

Deploy branch **`xiaole/stair-floors`** of `UniflexAI/tinynav` into a separate checkout.
It is based on `xiaole/stair-mode`, not the current `xiaole/rl-nav` branch.
Do not merge rl-nav, replace an unrelated checkout, erase databases, or reuse
another deployment's container name. This document does not require access to Spark.

The requested product flow is App -> Stairs -> DOWN/UP + relative floor count +
optional turn side -> search/align -> traverse flights and turn on landings -> stop.
One floor means two flights by default. DOWN 2 means four flights, not floor number 2
of a building. Start on a full-floor landing. Starting halfway down a staircase,
floor recognition from signs, and absolute building-floor localization are not implemented.

Validated on Spark, using Go2 MuJoCo, measured single-front depth, simulated odometry:

- DOWN 1: two flights, one turn, destination stop.
- DOWN 2: four flights, three turns, counts 1 -> 2 -> 3 -> 4, floors 0 -> 1 -> 2.
- Two-floor run: arrival at 120.5 s, 1.5 s zero-command hold, 7.5 mm horizontal drift.
- Minimum independent rectangular-footprint shaft clearance: 0.4954 m in the wider
  multi-floor scene; 0.1979 m in the narrower one-floor regression scene.
- No fall, wall/railing contact, or shaft footprint overlap in these runs.
- 25 focused tests pass after adding deployment configuration tests.

Latest x64 result (2026-11-21): full UP 9 and DOWN 9 pass in the ten-level,
1.2 m flight-width scene, with both 1.75 m and 1.2 m platform depths. The focused
stair/geometry suite passes 41 tests. These results supersede the earlier
first-flight failure; see [the regression report](stair_ten_level_narrow_trial.md).
Physical multi-floor operation remains unvalidated. The older stair-mode's
physical DOWN experience is not validation of these changes. Random moving doors,
real VIO, distributed DDS, and physical gait are outside this simulation result.

For the existing 3090 development server and interactive BEV simulation, see
[the development guide](stair_floors_development_simulation.md). For a physical
trial and the recording package to return, see [the field guide](stair_floors_physical_trial.md).

## 2. Architecture and boundaries

```text
External visual module / existing sensor driver
  depth + calibrated camera_info + synchronized VIO + raw high-rate VIO
       |
       v
  looper_bridge_node -> /slam/depth + /slam/odometry_visual + /slam/odometry
       |                         |
       v                         v
  stair_node                planning_node (retains occupancy integration)
  short-memory 2.5D BEV          ^
  geometry / tread plane        |
  search / landing turns -> /control/target_pose
  flight/floor counter          |
       |                        v
  /stair/status -> cmd_vel_control <- /planning/trajectory_path
                                  |
                               /cmd_vel
                                  |
             separately enabled Unitree interface / existing actuator bridge
```

Navigation is geometry/rules plus the existing local planner. There is no learned
navigation policy in this path. `go2_depth_v3.onnx` belongs to the optional MuJoCo
gait runtime; copying that file to this App does not install a physical-robot gait.
The repository's physical Unitree interface calls the robot's SportClient and
`ClassicWalk(True)`. Verify the actual robot/controller's stair capability separately.

Files to inspect when debugging:

| Purpose | Repository path |
| --- | --- |
| Floor count, pause, depth/odometry freshness, arrival latch | `tinynav/core/stair_node.py` |
| Height BEV, short point memory, tread alignment, landing-width estimate, search recovery | `tinynav/core/stair_target.py` |
| Occupancy, obstacle band, trajectory scoring, stair behavior | `tinynav/core/planning_node.py` |
| Path tracking, stale-path guard, blocked-status zero output | `tinynav/platforms/cmd_vel_control.py` |
| Body geometry, camera offset, speed limits | `tinynav/core/robot_specs.py` |
| Physical Unitree interface and NIC argument | `tinynav/platforms/unitree_control.py` |
| External module bridge and exact-time synchronization | `tool/looper_bridge_node.py` |
| App process lifecycle and environment configuration | `app/backend/node_manager.py` |
| REST contract | `app/backend/routers/nav.py` |
| Isolated two-floor simulation, capture and assertions | `tool/stair_mode_sim.py` |
| Bundled simulation geometry and camera calibration | `tool/stair_sim_course.py` |
| Targeted regression tests | `tests/test_stair_mode.py` |

The point memory in StairConfig defaults to 3 seconds; it is separate from planner
occupancy. Do not clear occupancy on every frame or label all unobserved cells free.
Near-body unobserved support can be inferred within `blind_radius` subject to
slope/height checks. That inference is an assumption, not a measured ground return.
The shaft checker in the MuJoCo harness uses world geometry only for independent
validation; it is not a live shaft sensor or a deployed safety supervisor.

## 3. Destination inventory (no motion)

Run these on the destination host and save the output:

```bash
mkdir -p "$HOME/tinynav-stair-handoff"
cd "$HOME/tinynav-stair-handoff"
{
  date -Is
  uname -a
  uname -m
  cat /etc/os-release
  lscpu
  free -h
  df -h .
  command -v docker || true
  docker version || true
  docker info || true
  nvidia-smi || true
  ip -br address
  docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  ss -lntup
} > inventory.txt 2>&1
cat inventory.txt
```

Expected architecture: `x86_64` / Docker `amd64`. The container uses Ubuntu 22.04,
ROS 2 Humble and Python 3.10. The host can use another Docker-supported Ubuntu
release; do not install Humble debs for a different host codename. Prefer the
container rather than changing the host Python or ROS installation.

For the full existing stack, provision an NVIDIA GPU with a working driver and
NVIDIA Container Toolkit. CPU-only full App deployment is not validated. The
Looper bridge, stair geometry and planner themselves do not require the stereo
TensorRT inference path, but App map lifecycle and other modules can still need
GPU models. Do not promise full CPU operation from an import-only check.

If Docker or GPU passthrough is missing, follow the official instructions matching
the actual Ubuntu release and installed driver. Do not replace an already working
Docker or driver installation blindly:

- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [ROS 2 Humble Ubuntu installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

After installing the toolkit, the standard rootful configuration is:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info
```

Restarting Docker affects other containers; account for existing workloads first.
Do not reuse Spark's ARM image export on x64 and do not set a rootless Docker
socket unless that is the destination's actual Docker setup.

## 4. Obtain the exact branch without disturbing other work

Host commands; authentication must use credentials already available on that host.
Do not embed tokens or passwords in the document, remote URL, or logs.

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ca-certificates
mkdir -p "$HOME/workspace"
cd "$HOME/workspace"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch xiaole/stair-floors --single-branch \
  https://github.com/UniflexAI/tinynav.git tinynav-stair-floors
cd tinynav-stair-floors
git lfs install --local
git status --short
git branch --show-current
git rev-parse HEAD
git log -6 --oneline
```

If the destination directory already exists, inspect it first. For a clean existing
checkout on this branch, use `git fetch origin` then `git pull --ff-only`.
Do not use `reset --hard`, delete local changes, or switch a live unrelated checkout.
The baseline functional commit is `156406f`; the deployment handoff commit also
contains this document, the Unitree environment switches and bundled sim course.
Record the complete deployed SHA in a local report; use that SHA for rollback.

Download ONNX models when using the RealSense perception path or App map features:

```bash
git lfs pull --include='tinynav/models/*.onnx' --exclude=''
git lfs ls-files
```

A small text file beginning `version https://git-lfs.github.com/spec/v1` is an LFS
pointer, not a usable model. Do not reuse Spark/Jetson `.plan` or `.so` binaries.
TensorRT plans must match the destination GPU, architecture and TensorRT runtime.

## 5. Image and isolated deployment container

Use the repository's existing image family, but verify the target platform:

```bash
docker pull --platform linux/amd64 uniflexai/tinynav:latest
docker image inspect uniflexai/tinynav:latest \
  --format '{{.Os}}/{{.Architecture}} {{json .RepoDigests}}'
docker run --rm --platform linux/amd64 --gpus all --entrypoint bash \
  uniflexai/tinynav:latest -lc 'uname -m; nvidia-smi; python --version'
```

Record the actual amd64 RepoDigest and use it as `TINYNAV_IMAGE` below.
In a new host shell, reload that variable from the saved `image.txt`. The tag
`latest` is mutable and a successful Spark ARM pull does not prove that an amd64
variant is available. If the amd64 manifest is unavailable, obtain an amd64 image
from the maintainer. Do not silently use emulation for a physical robot deployment.

A source Docker build is a separate fallback: the current Dockerfile assumes an
architecture-specific base image, `ARCH`, pinned TensorRT apt packages, and several
external source repositories. Review those against the target GPU/CUDA first;
`docker build --build-arg ARCH=...` alone does not fix the missing ARG declaration
or ensure the pinned packages exist. Do not describe an untested source build as
an equivalent, verified image.

Create a host environment file. Replace the image value with the verified digest.
Initially keep the physical robot interface disabled:

```bash
cd "$HOME/workspace/tinynav-stair-floors"
mkdir -p "$HOME/tinynav-stair-data"
cat > "$HOME/tinynav-stair-handoff/runtime.env" <<'ENV'
ROBOT_TYPE=go2
TINYNAV_DB_PATH=/tinynav_data
STAIR_CAMERA_HEIGHT=0.4669
TINYNAV_UNITREE_ENABLED=0
UNITREE_NETWORK_INTERFACE=enp3s0
ROS_DOMAIN_ID=0
ROS_LOCALHOST_ONLY=0
UV_PROJECT_ENVIRONMENT=/opt/venv
UV_NO_SYNC=1
OPENBLAS_NUM_THREADS=1
BACKEND_PORT=8000
FRONTEND_PORT=8080
ENV

docker image inspect uniflexai/tinynav:latest --format '{{index .RepoDigests 0}}' \
  > "$HOME/tinynav-stair-handoff/image.txt"
TINYNAV_IMAGE="$(cat "$HOME/tinynav-stair-handoff/image.txt")"
docker run -dit --name tinynav-stair-floors --platform linux/amd64 \
  --gpus all --network host --privileged --shm-size=4g \
  --env-file "$HOME/tinynav-stair-handoff/runtime.env" \
  -v "$HOME/workspace/tinynav-stair-floors:/tinynav" \
  -v "$HOME/tinynav-stair-data:/tinynav_data" \
  -v /dev:/dev \
  --entrypoint bash "$TINYNAV_IMAGE"
```

The privileged /dev access follows this repo's development-container setup and is
needed only where the chosen hardware driver requires it. For a network-only
external module, reduce device privileges after validating that driver. Host
networking exposes ports directly; check 8000/8080 availability before starting.
Use a trusted LAN: this App does not provide an authenticated public deployment.

`--entrypoint bash` deliberately bypasses the image's interactive engine-build
prompt. Model generation is an explicit step later. `/tinynav` is required because
App subprocess commands use this absolute path. Keep DB data on a separate mount.
`UV_NO_SYNC=1` prevents child `uv run` processes from unexpectedly rebuilding or
downloading dependencies during operation. Install/validate dependencies explicitly
before launching the App. A container recreation is needed after changing env-file
values; `docker restart` does not reload that file.

## 6. Validate the container environment and rebuild the frontend

Every following `docker exec ... bash -lc` must explicitly source ROS. A login
shell does not reliably source the image's interactive `.bashrc`.

```bash
docker exec tinynav-stair-floors bash -lc '
  set -e
  source /opt/ros/humble/setup.bash
  cd /tinynav
  python --version
  command -v python uv uvicorn flutter
  python -c "import numpy, scipy, numba, cv2, rclpy, cv_bridge, message_filters, fastapi, uvicorn; print(numpy.__version__)"
  python -c "from tinynav.core import stair_node, planning_node; from tinynav.platforms import cmd_vel_control; import app.backend.main; print(\"imports OK\")"
  python -c "from tinynav.core.robot_specs import ROBOT_CONFIG; print(ROBOT_CONFIG)"
'
```

Expected NumPy is 1.26.1 for the ROS cv_bridge ABI. Use the image's Python 3.10
venv with ROS system packages. Do not replace NumPy with 2.x or create a plain venv
that hides ROS packages. Check imports with the same Python that App subprocesses
will use, not just `/usr/bin/python3`.

If required packages or the native extension are missing, enter the container,
source ROS, then install the locked project explicitly:

```bash
cd /tinynav
uv sync --frozen --extra unitree
python -c 'from tinynav import tinynav_cpp_bind; print(tinynav_cpp_bind.__file__)'
```

The C++ build needs Ceres, Eigen and pybind11; see `tinynav/cpp/CMakeLists.txt`.
The image already provisions these. If `uv sync --frozen` reports a real dependency
failure, save its error and resolve that failure explicitly. Do not regenerate
`uv.lock`, upgrade the entire dependency graph, or copy an ARM extension as a fix.
Some environments expose a different native extension import location; inspect
the installed package and imports in the node that failed rather than guessing.

Rebuild Flutter Web from this checkout. The image's baked frontend can belong to
another branch, and bind-mounting the repo hides it:

```bash
docker exec tinynav-stair-floors bash -lc '
  set -e
  cd /tinynav/app/frontend
  flutter pub get
  flutter build web --release --suppress-analytics
  test -s build/web/index.html
'
```

For an offline browser, check the available Flutter version's support for
`--no-web-resources-cdn` and whether CanvasKit/fonts are bundled. Record the Flutter
version; do not silently upgrade it during a field deployment. Refresh the browser
cache after rebuilding. An old App UI can otherwise omit the new floor picker.

Run focused tests without external pytest plugin autoloading:

```bash
docker exec tinynav-stair-floors bash -lc '
  set -e
  source /opt/ros/humble/setup.bash
  cd /tinynav
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_stair_mode.py tests/test_stair_sim_course.py
'
```

If pytest is not installed in the selected venv, install it there explicitly.
The autoload flag avoids the known mixing of system pytest with a newer anyio
plugin (`ModuleNotFoundError: _pytest.scope`). Do not alter runtime packages just
to enable unrelated pytest plugins. Run both tests/test_stair_mode.py and tests/test_stair_sim_course.py; expect 41 tests in this revision.
The older backend suite also has four pre-existing failures on the base commit;
see `docs/stair_floor_trial.md` for their scope. Do not claim all tests pass.

### TensorRT models when the chosen path needs them

For the RealSense stereo perception path, build at least `superpoint`, `lightglue`
and `retinify`. Map/recognition features can require `dinov2` and SigLIP too. The
existing `all` target covers these six engines. Use actual LFS ONNX files and the
GPU/TRT tools in the target container:

```bash
docker exec tinynav-stair-floors bash -lc '
  set -e
  cd /tinynav
  nvidia-smi
  trtexec --version
  make -C tinynav/models all
  find tinynav/models -name "*_x86_64.plan" -size +1M -print
'
```

If trtexec is absent from PATH, inspect `/usr/src/tensorrt/bin/trtexec` and pass
`TRTEXEC=/actual/path/trtexec` to make. If a plan already exists for another GPU,
back it up and rebuild just that selected engine; `make` otherwise considers the
file up to date. Do not run `make clean` against an unrelated deployment's models.
Do not hand-copy generated plans across GPUs as proof of compatibility. The
Looper depth/VIO-to-stair pipeline does not call the stereo model, but App's map
restart after arrival can still exercise map models; inspect its log separately.

## 7. External visual module contract

The actual module driver/firmware and its launch command are NOT packaged here.
Use the destination's working vendor installation. Identify it from running
services, containers, existing launch files and the module documentation; do not
substitute a stock RealSense driver for a module that already provides depth/VIO.

Start that module before the App, in the same ROS domain and reachable DDS network.
The backend uses `/insight_full` in `ros2 node list` to identify the Looper path;
otherwise it launches the RealSense driver plus perception. If detection is wrong,
fix node discovery or restart after the sensor is ready. Do not run both pipelines
on the same output topics.

Required Looper input topics:

| Topic | ROS type | Requirement |
| --- | --- | --- |
| `/camera/camera/infra1/camera_info` | `sensor_msgs/msg/CameraInfo` | Actual calibrated left/depth intrinsics |
| `/camera/camera/infra1/image_rect_raw` | `sensor_msgs/msg/Image` | Left image synchronized with depth and VIO-image pose |
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | `32FC1` meters or `16UC1`/`mono16` millimeters |
| `/camera/camera/vio_image` | `geometry_msgs/msg/PoseStamped` | T_world_camera, exact same stamp as depth and left image |
| `/camera/camera/vio_100hz` | `geometry_msgs/msg/PoseStamped` | Raw high-rate T_world_camera, used for freshness/jump detection |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Calibration record and downstream compatibility |

The bridge uses `TimeSynchronizer`, not approximate-time matching. Equal frame
rates without identical timestamps are insufficient. Current subscriptions use
reliable sensor QoS; a best-effort-only publisher will need an explicit compatible
QoS adaptation. Check `ros2 topic info -v TOPIC` before changing synchronization.
The bridge aliases left CameraInfo to `/camera/camera/infra2/camera_info` because
the planner and stair node consume that topic. Do not supply RGB intrinsics there.

Outputs expected after the bridge starts:

- `/slam/depth`: `sensor_msgs/msg/Image`, `32FC1`, meters.
- `/slam/odometry_visual`: `nav_msgs/msg/Odometry`, synchronized to depth.
- `/slam/odometry`: `nav_msgs/msg/Odometry`, high-rate raw pose.
- `/camera/camera/infra2/camera_info`: left/depth CameraInfo alias.

Frames: camera optical +X right, +Y down, +Z forward; world Z up; body +X forward,
+Y left. Poses are camera optical poses, not unconverted base_link poses. Do not
apply the camera-to-body transform twice. A wrong axis or depth unit can produce
plausible images with unsafe geometry; inspect a stationary floor and wall first.

### Measured reference calibration

These are the supplied module measurements, not permission to overwrite a different
camera's live calibration:

| Camera | Width x height | fx, fy, cx, cy (pixels) | Horizontal x vertical FOV |
| --- | --- | --- | --- |
| Left / right IR | 544 x 640 | 303.10, 303.10, 269.37, 314.67 | 83.8 x 93.1 degrees |
| RGB | 1088 x 1920 | 762.90, 763.29, 540.69, 956.43 | 71.0 x 103.0 degrees |

Depth and left optical frames coincide. Right relative to left: +X 0.10076 m,
identity rotation. RGB relative to left: (0.05097, -0.00059, 0.00050) m,
xyzw quaternion (0.000374, 0.000924, -0.000012, 0.9999995).
Measured mount: forward 0.2709 m, right 0.0211 m, standing ground height 0.4669 m,
no downward pitch. Near-front floor can therefore be outside the image.

The simulator applies these measurements. The real planner's current Go2 defaults
are still camera_x=0.2, camera_y=0.0, body length=0.4, width=0.3, safety_radius=0.2.
`STAIR_CAMERA_HEIGHT` changes the stair reference height only; it does not override
all planner geometry. Compare the destination mounting/control center and actual
footprint with `robot_specs.py` before physical operation. If adjusting them,
record a small reviewed patch and repeat flat-ground/planner tests; do not claim
the current simulated clearance proves clearance with a different real footprint.

Inspect input/output while the robot interface remains disabled:

```bash
docker exec -it tinynav-stair-floors bash
source /opt/ros/humble/setup.bash
ros2 node list
ros2 topic info -v /camera/camera/vio_image
ros2 topic info -v /camera/camera/depth/image_rect_raw
ros2 topic echo /camera/camera/infra1/camera_info --once
ros2 topic hz /camera/camera/depth/image_rect_raw
ros2 topic hz /camera/camera/vio_100hz
```

`ros2 topic hz` runs until Ctrl-C. After App startup repeat the checks for
`/slam/depth`, `/slam/odometry_visual`, `/slam/odometry` and the aliased CameraInfo.
If using CycloneDDS, ensure `CYCLONEDDS_URI` names a real, appropriate XML file.
This branch does not contain the `scripts/cyclone_dds_localhost.xml` path inherited
by some images. Do not let a nonexistent XML or a loopback-only interface silently
isolate the external module. Use `unset CYCLONEDDS_URI` for default discovery or an
explicit tested interface configuration, consistently across the relevant processes.

## 8. Start App with robot interface disabled

Start only after the desired sensor is visible. The first terminal keeps the
container alive; the App uses its own detached tmux session:

```bash
docker exec -dit tinynav-stair-floors bash -lc '
  source /opt/ros/humble/setup.bash
  unset CYCLONEDDS_URI
  if [ -f /3rdparty/ros2_ws/install/local_setup.bash ]; then
    source /3rdparty/ros2_ws/install/local_setup.bash
  fi
  if [ -f /3rdparty/message_filters_ws/install/local_setup.bash ]; then
    source /3rdparty/message_filters_ws/install/local_setup.bash
  fi
  cd /tinynav
  /tinynav/scripts/start_app.sh
'
docker exec tinynav-stair-floors tmux list-sessions
curl --fail http://127.0.0.1:8000/device/status
curl --fail http://127.0.0.1:8000/sensor/mode
curl --fail http://127.0.0.1:8000/nav/stair/settings
```

The RealSense path also requires its built ROS overlay. Before App launch, source
`/3rdparty/ros2_ws/install/local_setup.bash` if present. If the image includes the
custom message_filters overlay, source
`/3rdparty/message_filters_ws/install/local_setup.bash` as well. Put these before
`start_app.sh` in the same shell so child processes inherit the overlay. A missing
vendor module driver still cannot be fixed by sourcing the RealSense overlay.

Only unset CYCLONEDDS_URI when not using a deliberate tested DDS config; preserve
such a config instead. The App's backend spawns sensor/planner children and the
Unitree interface unless disabled. `TINYNAV_UNITREE_ENABLED=0` disables that one
interface; it does not isolate another ROS subscriber on the same network.
For a computer-only test with no intended hardware input, use an isolated ROS
domain and no external actuator bridge. Do not assume changing ROS_DOMAIN_ID
isolates Unitree's own SDK channel, which uses DDS domain 0.

Open `http://DESTINATION_IP:8080` in the browser. In App setup, enter the actual
backend host/IP (default backend port 8000), not localhost on a separate phone.
The API schema is at `http://DESTINATION_IP:8000/docs`. HTTP 200 alone is not proof
that sensor/planner child processes started successfully; inspect DB logs.

If ports were changed, ensure the frontend's host/port handling matches. Avoid
HTTPS pages calling an HTTP backend, which browsers can block as mixed content.

## 9. App/API task configuration and behavior

The following commands can enable autonomous navigation output. Use them in the
isolated output-validation setup first; do not run against an unexpected live
actuator bridge. Check `/cmd_vel` subscribers before task start.

Persist measured height and a finite default count for the NEXT task:

```bash
curl --fail -X PUT http://127.0.0.1:8000/nav/stair/settings \
  -H 'Content-Type: application/json' \
  -d '{"camera_height":0.4669,"floors":1,"landings_per_floor":2}'
```

The settings file is `/tinynav_data/stair_settings.json`. Explicit saved
camera_height takes precedence over `STAIR_CAMERA_HEIGHT`; use null to fall back.
A count of 0 means unlimited. Per-run overrides do not overwrite saved defaults.
The App picker offers direction, floor count and AUTO/LEFT/RIGHT; inspect the
request JSON if an old cached frontend appears to ignore floors.

Start relative DOWN 2 floors:

```bash
curl --fail -X POST http://127.0.0.1:8000/nav/stair/start \
  -H 'Content-Type: application/json' \
  -d '{"direction":"down","turn":"auto","floors":2,"landings_per_floor":2}'
```

Use `floors:1` for the first physical-floor trial. Turn hints describe the U-turn
side relative to the incoming flight, not an absolute map direction. App startup
will activate navigation and launch stair_node plus cmd_vel_control as needed;
there is no separate manual `/nav/active` publication in the App workflow.

Check these topics and App status:

```bash
ros2 topic echo /stair/status
ros2 topic echo /cmd_vel
ros2 topic info -v /cmd_vel
ros2 topic info -v /planning/trajectory_path
```

Typical progression for DOWN 2:

```text
down search ... landings=0/4 floors=0/2 recovery=0
down ok ... landings=0/4 floors=0/2 recovery=0
down search ... landings=1/4 floors=0/2 recovery=0
down search ... landings=2/4 floors=1/2 recovery=0
down search ... landings=3/4 floors=1/2 recovery=0
down arrived ... landings=4/4 floors=2/2 recovery=0
```

The mode detects sustained pitched-to-level transitions, with minimum signed
height progress (default 1.0 m) since the last counted landing. It does not count
every flat patch or infer building floor labels. Nonstandard stair geometry or a
start at an intermediate landing needs a deliberate counting convention.

On arrival, stair_node clears the target and the controller publishes zero. The
App deactivates navigation, preserves the arrival status, stops stair_node and
restarts map_node for later ordinary navigation. The final stair status topic
may therefore disappear; the retained App status is the completion record.
The controller can stop publishing once navigation is inactive, so a continuing
stream of zeros is not the App arrival contract. The physical actuator must
honor a stop and have its own command-loss timeout. This branch's Unitree bridge
has no independent timer watchdog in its Twist callback; do not treat it as a
verified hardware safety watchdog.

Pause/resume preserve the task count:

```bash
curl --fail -X POST http://127.0.0.1:8000/nav/pause
curl --fail -X POST http://127.0.0.1:8000/nav/resume
```

Leave stair mode manually:

```bash
curl --fail -X POST http://127.0.0.1:8000/nav/stair/stop
```

A new `stair/start` resets the relative count. Stop/arrival followed by a new task
is not a continuation of the old task. `/nav/cancel` is for ordinary navigation
and does not replace `/nav/stair/stop`.

## 10. Physical Unitree connection

The handoff adds two App environment options, preserving old defaults for existing
users:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TINYNAV_UNITREE_ENABLED` | `1` | `0`, `false`, `no`: skip App-managed Unitree bridge; `1`, `true`, `yes`: enable |
| `UNITREE_NETWORK_INTERFACE` | `enP8p1s0` | Actual Linux NIC passed to `unitree_control.py --network-interface` |

Choose the NIC using `ip -br address` and `ip route get ROBOT_IP`. Do not copy the
Jetson interface name onto x64. Keep robot control via one owner: either the
App-managed Unitree interface or a pre-existing working controller, not both.
The direct script still supports its explicit CLI argument. The App requires
`ROBOT_TYPE=go2` (or the actual supported robot model) for the Unitree process.

To enable the App-managed interface after output-only checks, edit the host env
file to set `TINYNAV_UNITREE_ENABLED=1` and the correct NIC, then recreate this
specific container using the saved image digest and the same mounts. Before
recreation, stop the App and remove only this deployment's container:

```bash
docker exec tinynav-stair-floors tmux kill-session -t app
docker stop tinynav-stair-floors
docker rm tinynav-stair-floors
# Repeat section 5 docker run with the edited env file and pinned image.
# Repeat environment/inputs checks and section 8 startup.
```

Do not enable automatic task start on boot. A successful App load or battery
subscription does not prove SportClient command delivery or stair gait support.
Start with zero-output/stop and flat-ground directional checks, then a supervised
single DOWN flight/floor before any two-floor run. Keep the robot's existing
physical stop available. This document's deployment commands do not deploy the
MuJoCo ONNX gait onto the physical robot.

DDS details to check with the existing robot setup: SDK ChannelFactoryInitialize
uses domain 0 and the selected NIC; the subscriber is `rt/cmd_vel` with Unitree
DDS types. A normal ROS `/cmd_vel` publisher and a discovered battery topic are
not sufficient proof of end-to-end interoperability. Check the actual RMW, DDS
type compatibility, NIC routing and receive/stop behavior without duplicating
bridges. Do not globally change all DDS configuration to solve one missing topic.

## 11. Optional isolated MuJoCo regression on x64

Physical App deployment does not need MuJoCo. To reproduce the UP/DOWN tests, only
the clean tinynav-sim runtime checkout is external; the course and calibration
are now bundled in this branch. No rl-nav or rl-stairs checkout is required.

Host, using destination Git access:

```bash
cd "$HOME/workspace"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LooperRobotics/tinynav-sim.git
cd tinynav-sim
git checkout 7d63a42cdf71b4f672cfac3cda1ad6b26c3cfe3a
git lfs install --local
git lfs pull --include='mujoco/assets/policy/go2_depth_v3.onnx' --exclude=''
test -s mujoco/assets/go2/go2_pie.xml
```

The course builder supports missing optional visual meshes by removing mesh-only
visuals and retaining the training MJCF's inertials, joints and collision geoms.
The Spark GIF used that fallback. No 3DGS/model.zip bundle is needed for this test.
Verify the policy file is actual ONNX, not an LFS pointer. Access to the external
runtime repository and policy remains a separate prerequisite.

Prepare simulation dependencies in a separate host directory mounted into the
existing container. Tested versions on Spark: MuJoCo 3.14.0, onnxruntime 1.23.2,
glfw 2.11.0 and PyOpenGL 3.1.10. Install wheels for amd64 on the destination;
do not copy Spark's ARM wheel directory. To preserve the runtime environment,
use an explicitly mounted dependency directory in a disposable sim container:

```bash
mkdir -p "$HOME/tinynav-stair-sim/deps" "$HOME/tinynav-stair-sim/output"
docker run --rm --platform linux/amd64 --gpus all \
  -v "$HOME/tinynav-stair-sim:/simwork" \
  --entrypoint bash "$TINYNAV_IMAGE" -lc '
    set -e
    python -m pip install --no-deps --target /simwork/deps \
      mujoco==3.14.0 onnxruntime==1.23.2 glfw==2.11.0 PyOpenGL==3.1.10
  '
```

`--no-deps` protects the existing NumPy pin. Check dependency imports; install any
reported missing helper dependency into `/simwork/deps` without replacing NumPy.
If these exact wheels are unavailable for the selected Python/platform, report
that rather than silently testing a different MuJoCo release.

Run without network and without the App or physical Unitree interface. Use the
same image variable/digest as above:

```bash
docker run --rm --platform linux/amd64 --network none --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -e MUJOCO_GL=egl -e ROS_LOCALHOST_ONLY=1 -e ROS_DOMAIN_ID=79 \
  -e NUMBA_CACHE_DIR=/simwork/numba-cache \
  -e ROBOT_TYPE=go2 -e TINYNAV_UNITREE_ENABLED=0 \
  -v "$HOME/workspace/tinynav-stair-floors:/tinynav:ro" \
  -v "$HOME/workspace/tinynav-sim:/simrepo:ro" \
  -v "$HOME/tinynav-stair-sim:/simwork" \
  -w /tinynav --entrypoint bash "$TINYNAV_IMAGE" -lc '
    set -e
    source /opt/ros/humble/setup.bash
    unset CYCLONEDDS_URI
    export PYTHONPATH=/tinynav:/simwork/deps:${PYTHONPATH:-}
    python -c "import mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)"
    python -u tool/stair_mode_sim.py --sim-root /simrepo --floors 2 \
      --capture /simwork/output/down-two.npz
  '
```

Repeat with `--floors 1` and a different capture filename for one-floor regression.
First Numba compilation and rendering can take time. Expected final line begins
`PASS DOWN 2 floors`. The harness feeds the actual ROS node callbacks directly
inside one process and captures the actual planner result/controller commands;
it does not validate DDS communication between separate nodes or the App lifecycle.
Those require the App/ROS checks above. Never replace failed shaft assertions with
success flags or disable them to obtain a video.

Checks include every count transition against physical height; final destination
height and position; zero controller output during 1.5-second hold; no fall,
wall/railing contact, or footprint overlap with the shaft during traversal.
World course geometry is not fed to the navigation target generator.

NPZ fields: `detail`, `time`, `qpos`, `bev` (height, observed, reachable, planner
obstacle channels), `target`, `path`, `cmd`, `status`, `landings`, `clearance`,
`stop`, `hold`, `landing_events`, `floors`. The NPZ is capture data, not a GIF.
The delivered GIF was rendered separately; a GIF encoder is not bundled here.
An assertion failure can also save a diagnostic NPZ; file existence is not proof
of success. Require process exit 0, PASS log, four landing events and a valid hold.

## 12. Logs, recording, update and rollback

Child logs are in `/tinynav_data/logs/`, persisted as `$HOME/tinynav-stair-data/logs`
on the host. Inspect the most recent `looper_bridge`, `planning`, `stair_node`,
`cmd_vel_control`, `unitree` and `map_node` files. Check process exits as well as
HTTP responses. `docker logs` primarily shows the idle shell; App panes are in tmux:

```bash
docker exec tinynav-stair-floors tmux capture-pane -pt app:0.0 -S -120
docker exec tinynav-stair-floors tmux capture-pane -pt app:0.1 -S -60
docker exec tinynav-stair-floors ls -lt /tinynav_data/logs
```

Use App field recording, or the endpoints below, to capture sensor inputs, stair
status, targets, planner paths/occupancy and cmd_vel for a reproducible field issue:

```bash
curl --fail http://127.0.0.1:8000/field-record/info
curl --fail -X POST http://127.0.0.1:8000/field-record/start
# Execute the intended, supervised test.
curl --fail -X POST http://127.0.0.1:8000/field-record/stop
```

Record deployed commit, image digest, env (excluding credentials), camera_info,
TF, robot geometry, NIC, task JSON and actual landing count. Preserve bags under
the DB `debug/field_*` folders. Check free disk space before long recordings.

For an update: stop the current task; ensure the robot is stationary; stop App;
back up `stair_settings.json` and save the old SHA/digest. In the clean checkout,
`git fetch origin` and `git pull --ff-only`; rerun tests and rebuild Flutter; restart
App. Recreate the container when environment/image changes. Do not edit files under
a running controller and assume its processes reload automatically.

For rollback, use a separate checkout of the saved SHA and a separate container
or restore this clean checkout while stopped. Retain the DB backup and image
digest. Do not run old and new command publishers concurrently.

For a network-restricted destination, prepare on an amd64 staging machine:
Git bundle/source checkout with the selected commit, actual LFS ONNX objects,
`docker save` of the verified amd64 image, rebuilt frontend, required amd64 wheels,
and optional tinynav-sim checkout/policy. Generate SHA256 checksums and verify after
transfer. A Git bundle alone omits LFS objects, container layers and optional assets.
Do not export Spark's ARM image or wheel cache as an x64 installation package.

## 13. Troubleshooting by observed symptom

| Symptom | Check / next action |
| --- | --- |
| `exec format error` | Host/image/native-library architecture; require amd64 image and x64 wheels |
| Docker cannot select GPU runtime | Host driver, Container Toolkit, Docker runtime; test nvidia-smi inside image |
| App opens but floor picker is missing | Branch SHA, rebuilt Flutter `build/web`, browser/service-worker cache |
| Backend starts RealSense with a Looper module | `/insight_full` discovery, ROS domain, multicast, startup order; restart after module ready |
| Raw depth exists but `/slam/depth` does not | Bridge log, exact 3-topic timestamps, CameraInfo and reliable QoS |
| `depth_invalid` | Depth freshness >0.75 s; matched pose timestamps; do not bypass the guard |
| `odom_invalid` | VIO jumps/gaps, axis convention and raw pose continuity; preserve the stop/recovery logic |
| Targets exist but cmd_vel stays zero | `/nav/active`, pause, blocked stair status, planner paths, path age and node logs |
| cmd_vel exists but robot does not move | Disabled interface, wrong NIC, SDK domain/types, robot control mode, another command owner |
| Wrong count after a landing | Starting level, landings_per_floor, actual flight height, body pitch and VIO drift; inspect bag, do not just lower thresholds |
| Immediate unlimited traversal | `floors=0` in saved settings or an old client omitting the per-run override |
| Arrival topic disappears | App intentionally stops stair_node; inspect retained stairStatus and nav inactive state |
| TensorRT engine deserialize failure | Rebuild target GPU/TRT engine from actual ONNX; do not reuse ARM/other-GPU plans |
| `_pytest.scope` import failure | Disable pytest plugin autoload and use a compatible pytest in the selected venv |
| Ceres/Eigen/native extension import failure | Correct Python/architecture and explicit locked native build; retain full compiler error |
| EGL initialization failure in simulation | GPU graphics capability, libEGL, driver access; no DISPLAY needed for EGL |
| Sim policy parsing failure | LFS pointer vs actual ONNX; exact tinynav-sim commit and policy hash |
| User sees a plausible BEV but unsafe target | Check real depth units, intrinsics, optical/world axes, footprint, blind support and source frames |

Do not modify the no-shaft-overlap assertions, memory, safety radius or drop guards
simply to make a failed deployment test pass. Preserve evidence and isolate the
actual sensor/geometry/controller error.

## 14. Copy this prompt into the destination GPT window

```text
Deploy TinyNav from UniflexAI/tinynav branch xiaole/stair-floors on this Ubuntu x64
computer. Read AGENTS.md and docs/stair_floors_ubuntu_x64_deployment.md completely
before acting, and treat the document as the handoff for this branch.

Inspect this computer's actual environment, existing containers, GPU, ROS, camera
module and robot network. Use a separate checkout/container/data directory. Keep
other navigation work and existing deployments intact. First validate environment,
frontend/API, sensor contract, target/planner output and focused tests with the
App-managed Unitree interface disabled. Do not start a physical stair task as part
of software installation. Use the explicit NIC/enable environment options when
later connecting the intended physical controller; avoid duplicate command owners.

Implement the deployment and resolve concrete failures rather than only explaining
commands. Obtain the correct amd64 image and actual model assets; do not invent
missing vendor driver commands, transfer ARM binaries, remove safety checks, or
claim simulation proves field readiness. If an external prerequisite is missing,
identify its exact name and continue all independent checks. The bundled MuJoCo
course does not need the rl-nav/rl-stairs branch; optional runtime instructions are
in section 11. Full UP 9 and DOWN 9 are validated only in isolated simulation; physical trials remain pending.

Write a local deployment report with commit SHA, image digest, paths, runtime env,
startup/restart/stop commands, camera/NIC settings, checks and their real outputs,
remaining blockers and rollback steps. Do not publish credentials in that report.
Explain results to me in Chinese.
```


## 15. Current narrow ten-level result

The 1.2 m flight-width scene now passes full UP 9 and DOWN 9, at platform depths
1.75 m and 1.2 m. [The trial report](stair_ten_level_narrow_trial.md) records the
initial shaft failure, entry-alignment fix, final-platform fix and retained
assertions. Run both focused suites (41 tests). [The simulation guide](stair_floors_development_simulation.md)
has exact commands, runtime pin and interactive BEV instructions.

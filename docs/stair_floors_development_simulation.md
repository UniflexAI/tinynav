# Stair-floors development and simulation

For this App/stairs-mode update, including the existing-device upgrade steps,
see [App upgrade and use (Chinese)](stair_app_upgrade_zh.md). Updating Python
sources and restarting backend/navigation processes is required in addition to
rebuilding the ordinary Flutter frontend.

Updated: 2026-11-21. Branch: `xiaole/stair-floors`.

## Scope and tested environment

The 3090 host is a development/simulation installation, not a robot deployment.
Host Ubuntu 20.04; container Ubuntu 22.04, ROS 2 Humble, Python 3.10.12,
NumPy 1.26.1. Base amd64 image:
`uniflexai/tinynav@sha256:96f802435261b774da2a72d7c965aa50137195491adc295a345bde0b0c06afa2`.
The installed derived image reference is saved in
`/home/dm/tinynav-stair-dev/dev-image.txt`; preserve it for the existing host.
MuJoCo 3.14.0, onnxruntime 1.23.2, glfw 2.11.0, PyOpenGL 3.1.10;
interactive viewer: viser 1.1.1 and trimesh 4.11.1.

Simulation runtime: `LooperRobotics/tinynav-sim`, pinned commit
`7d63a42cdf71b4f672cfac3cda1ad6b26c3cfe3a`, including the real LFS policy
`mujoco/assets/policy/go2_depth_v3.onnx`. The gait is ONNX in MuJoCo;
TinyNav supplies geometry targets, the local planner and velocity tracking.
Target callbacks run at 2 Hz, planning/control callbacks at 10 Hz in this harness;
policy runs at 50 Hz. The App's real command timer is separate.
Navigation uses simulated calibrated front depth and odometry. World geometry
only supplies rendering and independent safety assertions.

For a new machine, follow sections 3-6 and 11 of
[the x64 deployment guide](stair_floors_ubuntu_x64_deployment.md), including native
extension/environment checks and LFS assets. The base image alone is not the
complete tested development environment. Preserve NumPy and build native/TRT
assets for the actual architecture/GPU. Frontend rebuild is needed for App use,
not for the standalone simulation viewer.

## Existing 3090 paths and commands

- Source: `/home/dm/workspace/tinynav-stair-floors`.
- Runtime: `/home/dm/workspace/tinynav-stair-sim-runtime`.
- Host helpers/env: `/home/dm/tinynav-stair-dev`.
- Captures/dependencies: `/home/dm/tinynav-stair-sim`.
- Dev container: `tinynav-stair-dev`; live viewer: `tinynav-stair-live`.

The containers use bridge networking, `ROS_DOMAIN_ID=79`,
`ROS_LOCALHOST_ONLY=1`, `TINYNAV_UNITREE_ENABLED=0`. Do not attach a robot bridge
to this environment. Existing unrelated services remain separate.

```bash
/home/dm/tinynav-stair-dev/enter.sh
/home/dm/tinynav-stair-dev/test.sh
/home/dm/tinynav-stair-dev/sim-compact.sh down 9
/home/dm/tinynav-stair-dev/sim-compact.sh up 9
/home/dm/tinynav-stair-dev/live-start.sh
/home/dm/tinynav-stair-dev/live-stop.sh
```

These are installed host helpers, not repository scripts. Portable commands in
a prepared isolated container, with ROS sourced and simulation deps on PYTHONPATH:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_stair_mode.py tests/test_stair_sim_course.py tests/test_stair_odom_noise.py tests/test_stair_sim_control.py
python -u tool/stair_mode_sim.py --sim-root /simrepo --direction down \
  --levels 10 --floors 9 --flight-width 1.2 --landing-depth 1.2 \
  --capture /simwork/output/compact-down-nine.npz
python -u tool/stair_mode_sim.py --sim-root /simrepo --direction up \
  --levels 10 --floors 9 --flight-width 1.2 --landing-depth 1.2 \
  --capture /simwork/output/compact-up-nine.npz
```

Repeat with `--landing-depth 1.75` and unique output filenames for the original
platform regression. Ten platform levels mean nine floor changes, eighteen flights.
A shorter `--floors 1` or `2` task can use the same `--levels 10` building.
A pass requires process exit 0, PASS log, all expected ordered count transitions,
correct final pose, zero output and hold, positive shaft clearance, no contact/fall.
An NPZ or video alone is not proof of pass. Failed runs save diagnostic data with
`arrived=False`, including when the node reported arrival but final assertions failed.
Full results and failure analysis: [trial report](stair_ten_level_narrow_trial.md).

## Integrated TinyNav App

Open **http://192.168.11.116:18083/** -> **Operate** for the original TinyNav App's
front camera, local BEV, 3D occupancy and Stairs task buttons. See
[original App connection](stair_tinynav_app_view.md).
The observer/control page **http://192.168.11.116:18080/** provides environment
reset/pause, noise/speed settings and third-person views with BEV/depth. The same
world is available separately at `:18082`.
This isolated App simulation profile reuses the tested deterministic navigation
harness; it does not start physical sensors or hardware processes. Setup, control
semantics and execution limits: [App + MuJoCo guide](stair_mujoco_app.md).

## Interactive viewer and live BEV

The viewer source is now tracked as `tool/stair_live_sim.py` and
`tool/stair_bev_view.py`. It uses the runtime's `mujoco/webviewer.py`.
In the prepared container, install viewer dependencies into a separate directory:

```bash
python -m pip install --no-deps --target /simwork/viewer-deps viser==1.1.1 trimesh==4.11.1
export PYTHONPATH=/tinynav:/simwork/deps:/simwork/viewer-deps:${PYTHONPATH:-}
export TINYNAV_SIM_ROOT=/simrepo
export TINYNAV_SIM_OUTPUT=/simwork/output/live
python -u tool/stair_live_sim.py
```

Check viewer imports and add any missing helper dependencies to that separate directory
without replacing the runtime NumPy 1.26.1 pin.
Use EGL and the same isolation variables as the headless run. Bind container port
8012 to a trusted-LAN host port (existing host: 18012). For the existing live
container, change its launch command from `/devwork/live_sim.py` to
`/tinynav/tool/stair_live_sim.py` on recreation to use the versioned source.
The old installed helper remains compatible with the tested version.

Open `http://SERVER_IP:18012`, select `Ten-level narrow`, direction `up`/`down`,
floors `9`, and `Narrow landing depth (m)` of `1.2` or `1.75`. Press `Start / Reset`
to apply. The default is paused; `Pause physics` pauses/resumes, `STOP` pauses,
and Start resets the world/count. The camera supports orbit/zoom/pan.
Live panels show robot-centred heading-up 5 x 5 m stair height/observed ground,
TinyNav planner height map, obstacles/path/target and measured front depth.
No fake ground-truth free-space map is substituted for sensed BEV.

Interactive contact/fall/shaft assertion failure latches a pause and requires reset.
This independent world check is not a deployed sensor safety supervisor. The
headless harness additionally verifies count heights, final position and stop hold;
use it for regression qualification rather than relying on a viewer screenshot.

## Odom perturbation experiments

See [the odometry noise experiment](stair_odom_noise_trial.md) for deterministic
position/yaw/pitch injection, CLI parameters, failure traces and validation scope.
Default simulation inputs remain unperturbed.

The final-support noise fix passes 52 focused tests, the reproduced position-only
DOWN 1 case, combined-noise UP/DOWN 9 and noiseless UP/DOWN 9. See the noise trial
report for exact settings, margins and single-seed limitations.

The live pause/resume parity repair and independent arrival/hold validation are
documented in [live/offline parity](stair_live_offline_parity.md). The four focused
suites pass 57 tests.

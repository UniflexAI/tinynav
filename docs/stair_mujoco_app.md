# TinyNav App + MuJoCo development environment

For the **original TinyNav App camera/local-map view**, use
**http://192.168.11.116:18083/** -> **Operate** and follow
[the original App connection guide](stair_tinynav_app_view.md). The page described
below at `:18080` is the separate MuJoCo observer/control page, not the original
Operate view.

The observer simulation profile controls one live MuJoCo instance, rather than playing a
recording. Open **http://192.168.11.116:18080/** on the server's LAN. The backend is
`:18081`, and the same 3D scene can be opened separately at
**http://192.168.11.116:18082/**. The earlier independent viewer at `:18012` is unchanged.

## Using the page

1. Select 1–9 floor changes, a platform depth (1.2 / 1.75 m), simulation speed
   (1 / 2 / 4), and odometry noise on/off. The staircase has ten full-floor platforms,
   eighteen flights, and 1.2 m flight width. Each floor change counts two landings. Speed is a pacing multiplier; achieved wall-clock
   speed also depends on rendering and CPU load.
2. Press **发起上楼任务** or **发起下楼任务**. Starting constructs the corresponding
   signed-height building, resets robot, odometry noise, target generator, planner,
   controller, counters and trace, and starts physics. The request acknowledges the
   new run ID only after the simulation loop has applied it.
3. **暂停仿真 / 继续仿真** freeze/resume simulated time. **停止任务** cancels the task;
   continuing a cancelled, failed or validated run is rejected. Start or reset again.
4. **重置环境（暂停）** applies the selected configuration to a fresh paused world.
   The reset keeps the current up/down direction. A task can be reset while running,
   except during a latched VIO fault: confirm recovery in Operate first.
5. **自由视角** permits mouse orbit, wheel zoom and right-drag pan. **跟随机器狗**
   moves the third-person camera with the robot. **全景** positions the camera farther
   away at the robot's current level. These controls do not change physics.
6. The embedded scene includes **TinyNav planner BEV (live)** and **Front measured
   depth**. At initial reset the depth is rendered immediately; the planner map fills
   after navigation callbacks start. App status shows the applied noise configuration,
   elapsed simulation time, ordered landing count and independent validation result.

Settings above the task buttons are pending until Start/Reset. Applied settings are
shown separately below the task status and in the scene. Noise defaults to combined,
scale 0.5, seed 7 (the previously tested noise trial). Its scale affects the base
parameters reported in state; see [noise model](stair_odom_noise_trial.md).

## Execution and isolation

`app/frontend` is the existing Flutter TinyNav App, built with
`TINYNAV_SIMULATION=true`. Its simulation page calls the FastAPI simulation profile.
That profile never starts the robot `NodeRunner`, physical sensors, mapping processes
or Unitree SDK. Unsupported robot routes are absent in this profile.

`tool/stair_live_sim.py` runs the real `StairNode`, `PlanningNode` and
`CmdVelControlNode` methods with calibrated simulated depth and estimated odometry.
Callbacks are wired directly in one process, with a deterministic simulation clock:
2 Hz targets, 10 Hz depth/planning/tracking, 50 Hz Go2 ONNX policy and physics control.
This is a closed-loop development harness, **not a distributed ROS/DDS or full SLAM
integration test**. Speed changes wall-clock pacing, preserving callback ordering.
MuJoCo produces depth; odometry starts from ground truth and optionally adds the
specified synthetic error. The environment is not a prebuilt SLAM map.

HTTP control threads enqueue commands; only the physics loop mutates the world and
navigation nodes. The internal control server binds `127.0.0.1:8013`. The simulation
container uses Docker bridge networking, ROS domain 80, localhost-only discovery,
Unitree disabled, no hardware devices or host network. The original development
containers/services are separate. Walls/rails are translucent in the copied render
model; collision geometry and physical safety checks remain unchanged.

Falls, wall/rail contact, body overlap with the shaft, ordered counts and their
heights, final landing pose, zero command, and 1.5 s stop-drift checks remain enabled.
A node's `arrived` message becomes a validated success only after the final hold.
Per-run JSON/NPZ evidence is saved at
`/home/dm/tinynav-stair-sim/output/app-live/runs/`, including interrupted runs when
reset/replaced. State is at `output/app-live/live-state.json`.

## Build and start on the prepared 3090

Keep the environment and pinned runtime described in
[development setup](stair_floors_development_simulation.md). Copy frontend sources to
`/simwork/app-profile` inside the dev container, then:

```bash
cd /simwork/app-profile
flutter pub get --offline
flutter build web --no-pub --no-web-resources-cdn \
  --dart-define=TINYNAV_SIMULATION=true \
  --dart-define=TINYNAV_DEFAULT_DEVICE=192.168.11.116:18081
```

The checked-in OFL font subset provides Chinese UI glyphs without fetching fonts
from a CDN. `tool/subset_stair_ui_font.py` regenerates it from upstream
Google Fonts `ofl/notosanssc/NotoSansSC[wght].ttf`; keep `OFL.txt` alongside the subset.

On this prepared host (ports must be free):

```bash
docker run -d --name tinynav-mujoco-app --restart unless-stopped \
  --entrypoint /bin/bash --gpus all \
  --env-file /home/dm/tinynav-stair-dev/dev.env \
  -e ROS_DOMAIN_ID=80 \
  -e TINYNAV_APP_WEB_ROOT=/simwork/app-profile/build/web \
  -e TINYNAV_SIM_OUTPUT=/simwork/output/app-live \
  -p 18080:8080 -p 18081:8000 -p 18082:8012 \
  -v /home/dm/workspace/tinynav-stair-floors:/tinynav:ro \
  -v /home/dm/workspace/tinynav-stair-sim-runtime:/simrepo:ro \
  -v /home/dm/tinynav-stair-sim:/simwork \
  -v /home/dm/tinynav-stair-dev:/devwork \
  "$(cat /home/dm/tinynav-stair-dev/dev-image.txt)" \
  -lc 'source /devwork/env.sh; exec bash /tinynav/scripts/run_mujoco_app.sh'
```

The separate scene page also has a collapsible **Scene controls** section. Its
**Reset environment (paused)** button uses the currently applied configuration
and updates the App's shared state. Its camera dropdown supports free orbit,
following and overview. Both pages operate on the same world.

For an existing installation use `docker restart tinynav-mujoco-app` after Python
changes, and rebuild the separate Flutter output after Dart/asset changes. Browser
refresh reconnects to the current run, without resetting physics. Multiple tabs see
and control the same shared instance. This page does not create per-user worlds.

To stop this profile only: `docker stop tinynav-mujoco-app`.

## Validation on 2026-11-21

Both tasks were launched through the App buttons with combined noise, scale 0.5,
seed 7, 1.2 m flights/platforms, ten full-floor platforms, and pacing speed 4.

| Task | Ordered landings | Elapsed simulation time including hold | Minimum shaft clearance | Result |
| --- | --- | --- | --- | --- |
| UP 9 | 1–18 | 370.02 s | 0.08903 m | Validated |
| DOWN 9 | 1–18 | 379.52 s | 0.05717 m | Validated |

UP included a mid-run pause, camera orbit/follow changes and browser refresh.
The first 3686 UP / 3781 DOWN samples exactly matched the prior offline reference
for time, true/estimated poses, noise, targets, commands and counts. App reset
while running and the scene page's reset each produced a new run ID, zero count,
zero command and paused initial time. Stop rejected a subsequent resume (409).
Invalid settings were rejected without replacing the current world; live planner
and measured-depth JPEG endpoints returned real frames.

Focused Python tests: **68 passed**. Flutter address/WS tests: **2 passed**.
Web release build succeeded with local CanvasKit/fonts. Flutter analysis reported
zero errors/warnings and 37 existing informational lints. All eleven original
headless safety assertions remain AST-identical to `efc25a2`; live assertions
remain active. Machine-readable run evidence and deployed fingerprints:
[validation JSON](validation/stair_mujoco_app_2026_11_21.json).


## Injecting VIO faults and recovering in Operate

The simulation now runs the same VIO guard class and ROS callback logic as the
physical backend. Original App Operate displays its lock/ready state and sends
an explicit fault-ID confirmation. A normal simulation Resume only unfreezes
physics; it cannot acknowledge the VIO latch. Locking clears both local maps and
old targets. Reset/Start and manual drive cannot bypass a lock. A fresh simulation
world has an in-memory guard; physical deployment uses a persistent state file.

For a controlled simulation-only injection, start an ordinary noisy stair task,
then call the simulation backend (never a physical deployment):

```bash
curl -X POST http://192.168.11.116:18081/simulation/vio-fault \
  -H 'Content-Type: application/json' -d '{"kind":"restart","duration":2}'
# kind also accepts jump (+2 m estimated X) and dropout (odom/depth silence).
curl http://192.168.11.116:18081/nav/vio
```

Wait for the Operate recovery banner to enable **Confirm VIO recovery**, inspect
the environment, then explicitly confirm. Nothing automatically clicks it in the
web application. Independent shaft/contact/count/stop assertions stay enabled;
a failed assertion still stops the simulation and its evidence is retained.
See [fault regression and physical deployment](stair_vio_safety.md) for exact
thresholds, test scope and the distinction between software stopping and physical
SDK/robot stopping distance.

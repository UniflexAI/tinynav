# Original TinyNav App view connected to MuJoCo

Use **http://192.168.11.116:18083/**, then open **Operate**. This is the existing
Flutter `HomePage` / `OperateTab`, with its own camera widgets and planning painters.
It does not embed the MuJoCo viewer. The earlier page at `:18080` remains an observer
and simulation configuration/reset tool; the separate third-person scene at `:18082`
shows the same world. All three pages control/observe one shared simulation instance.

## Use the App

- In **Operate**, the top camera defaults to **color**: the simulated robot's front
  camera. Its original topic selector switches to **depth**, the measured front depth.
  **Default** streams a maximum 320-pixel edge; **High** uses the 640-pixel source.
- The original local map displays ESDF, obstacles, the current target, selected path
  and footprint. Its **2D / 3D** button switches to the observed occupancy points.
  The cloud is sampled to at most 2500 points, using the planner's occupied-cell
  threshold. These are perception outputs, not the complete scene mesh.
- Press the original **Stairs** button, select **1–9 floors**, then **Up / Down**.
  Two flights count as one floor. Starting a task creates the corresponding fresh
  up/down scene and resets robot/navigation state. Turn selection is Auto for this
  validated course. The ordinary device's original choices remain unchanged.
- The original **Pause / Continue**, active **Stairs** stop, **Nav ON** toggle and red
  **STOP** operate this simulated task. STOP cancels the task and freezes physics.
  Start a new task or reset after cancellation; a cancelled task cannot resume.
- Use `:18080` or the **Scene controls** section at `:18082` to reset or change the
  third-person camera. A reset immediately replaces the App's camera/map state too.
  The observer's applied noise, platform depth and speed are reused by App task starts.
  Pending observer form selections do not apply until that page starts/resets a world.

The App labels this connection **MuJoCo SIM / Local perception**. The current
harness uses simulated front depth plus odometry (optionally noisy); it does not run
full image-based SLAM/global map localization. `/map/current` therefore returns 404,
`localized` stays false, and the global POI list is empty. The local map remains live.
Manual sticks are enabled with a 0.5-second deadman timeout; manual input cancels
autonomous navigation. Physical sit/stand actions remain unavailable for this gait.
The Stairs settings page also saves 1–9 floors and offers **Start upstairs** /
**Start downstairs**. Two flights per floor and camera calibration are fixed for
this simulated building. Red STOP remains available. Field rosbag recording is hidden; per-run JSON/NPZ traces
remain at `/home/dm/tinynav-stair-sim/output/app-live/runs/`.

Stairs mode uses live depth and estimated odometry without a prior global map.
Online mapping and POI navigation are outside this profile. The current simulator
uses measured MuJoCo depth and configurable noisy simulated odometry.

## Data and command wiring

`tool/stair_app_bridge.py` reuses the original `BackendNode` serializer methods,
without constructing `BackendNode` or starting its physical-device lifecycle.
Actual outputs of the existing simulated `StairNode`, `PlanningNode` and
`CmdVelControlNode` populate the original App contracts:

| App interface | Live source |
| --- | --- |
| `/ws/preview`, color | MuJoCo front RGB camera, same calibrated viewpoint as depth |
| `/ws/preview`, depth | Measured optical-Z depth, fixed 0-5 m display palette |
| `/ws/planning`, ESDF/mask/grid bounds | Published planner height-map and obstacle-mask messages; original PNG transforms |
| `/ws/planning`, target/path/footprint | Actual stair target, selected trajectory and planner footprint |
| `/ws/planning`, 3D points | Planner occupancy grid reconstructed from depth + estimated pose |
| `/ws/pose`, odometry | Same estimated camera pose supplied to navigation, including applied noise |
| `/ws/status` | Actual task direction/status/pause/cancel/validation state |
| `/nav/stair/start` | Original App's Stairs request -> loop-queued fresh simulation task |
| `/nav/pause`, `/nav/resume`, `/nav/stair/stop` | Same simulation instance's pause/resume/cancel |

The simulation clock, node callback order, gait policy and independent safety
assertions remain unchanged. The transport is the deterministic in-process harness,
not a distributed ROS/DDS or full-SLAM integration qualification. The App receives
structured data and renders its existing views; rendered observer screenshots are
not supplied as the App's local map.

## Build and serve on the prepared 3090

Keep the backend/simulation service from [the MuJoCo App guide](stair_mujoco_app.md).
Copy `app/frontend` into `/simwork/native-app-profile` in `tinynav-stair-dev`, then:

```bash
cd /simwork/native-app-profile
flutter pub get --offline
flutter build web --no-pub --no-web-resources-cdn \
  --dart-define=TINYNAV_DEFAULT_DEVICE=192.168.11.116:18081
```

Do **not** pass `TINYNAV_SIMULATION=true` for this build: that flag selects the
separate observer page. The same frontend source retains both views.

The installed static UI container is:

```bash
docker run -d --name tinynav-stair-app-ui --restart unless-stopped \
  --entrypoint /usr/bin/python3 -p 18083:8080 \
  -v /home/dm/tinynav-stair-sim/native-app-profile/build/web:/web:ro \
  "$(cat /home/dm/tinynav-stair-dev/dev-image.txt)" \
  -m http.server 8080 --bind 0.0.0.0 --directory /web
```

The static UI container has no GPU/hardware mounts. The backend at `:18081` remains
isolated with Unitree disabled and ROS localhost-only/domain 80. Rebuild Flutter
after Dart changes; restart `tinynav-mujoco-app` after Python changes. Refresh the
App after a backend restart to reconnect its WebSockets.

## Validation on 2026-11-21

Both full tasks were started by clicking the original App's **Operate -> Stairs ->
Up / Down** controls, with nine floors selected. Noise was combined / scale 0.5 /
seed 7, flight and platform widths/depth 1.2 m, and simulation pacing 4.

| Original App task | Ordered landings | Elapsed simulation including hold | Minimum shaft clearance | Result |
| --- | --- | --- | --- | --- |
| UP 9 | 1-18 | 370.02 s | 0.08903 m | Passed |
| DOWN 9 | 1-18 | 379.52 s | 0.05717 m | Passed |

The first 3686 UP and 3781 DOWN samples were exactly equal to the prior offline
reference for time, true/estimated poses, applied noise, target, command and count.
All independent safety assertions stayed enabled. A pause through the original
App held pose/time fixed and command zero; Continue completed the same UP run.

Browser verification covered the original color/depth topic selector, Default/High
selector, 2D/3D local perception toggle, native Stairs dialogs and arrival state.
Live stream checks confirmed full 640x544 RGB/depth sources, metric grid origin/
resolution, actual footprint/trajectory and a bounded observed voxel cloud.
The UP run used the initial 320x272 RGB preview implementation; the DOWN run also
qualified the subsequent 640x544 RGB/high-quality preview implementation.

Focused Python tests: **72 passed**, including original App WS contracts and
original grid serialization/orientation. Flutter URL/WS tests: **2 passed**.
Release web build passed; analysis had zero errors/warnings and 38 informational
lints. Full per-run evidence and deployed source hashes:
[validation JSON](validation/stair_native_app_2026_11_21.json).

## Stairs-mode control regression (2026-11-21)

The selected scope is stairs mode without scanning a global map or POI navigation.
The separate full-SLAM experiment was stopped and its source/logs archived; it is
not the default App runtime. The default profile uses rendered measured depth and
noisy simulated odometry, and the original stair/planning/control node methods.

After restoring that profile and fixing manual control/task settings, both tasks
were launched from the original App again. DOWN used **Stairs settings -> Start
downstairs**; UP used **Operate -> Stairs -> Up**. Both used nine floors, two flights
per floor, 1.2 m flight/platform dimensions, combined noise / scale 0.5 / seed 7.

| Task | Ordered landings | Task simulation time including hold | Minimum shaft clearance | Result |
| --- | --- | --- | --- | --- |
| DOWN 9 | 1–18 | 379.52 s | 0.05717 m | Passed |
| UP 9 | 1–18 | 370.02 s | 0.08903 m | Passed |

Final commands were zero and the independent 1.5-second stop hold passed. All
physics safety assertions remain enabled; the eleven headless assertion ASTs are
unchanged from `efc25a2`. The current targeted test selection passed 68 Python tests
and three Flutter tests; both original-App and observer web builds succeeded.
This is a simulator regression, not visual-SLAM odometry or real-robot validation.

App manual input has a 10 Hz heartbeat and a 0.5-second deadman timeout. Disconnect
sends zero even if the WebSocket task is cancelled. A browser stick drag was
confirmed to move the simulated robot, followed by zero output on release. Manual
input cancels the stair task; starting a new stair task resets the environment.
The original App's front depth and BEV were checked during the full runs.

Structured results: [stairs-mode control validation](validation/stair_app_controls_2026_11_21.json).
Server traces remain in `/home/dm/tinynav-stair-sim/output/app-live/runs/` with run IDs
`down-9-1795262249213823108` and `up-9-1795262532614228553`.

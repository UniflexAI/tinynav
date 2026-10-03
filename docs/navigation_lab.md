# Navigation lab: Jev integration stages 1–2

Run `bash scripts/run_ros_planning_web.sh` inside the ROS container and open port 8766.
No Jev API requests are made in these stages.

## Repeatable baselines

Choose a scenario and click **Run baseline**. This uses fixed GO2 defaults,
not the edited robot fields, and resets pose, observation memory, planner and
simulator control. The scenario catalog is shared by the web UI and batch runner.
Empty, obstacle bypass (Open target), L turn and Dead end escape are the primary
cases; existing Straight, S bend, Narrow gate and Back target remain available.

Run a batch from the repository root while the web server is available:

```bash
python3 scripts/run_navigation_baseline.py --cases empty open_target l_turn dead_end --repeats 3 --timeout 120
```

Only one operator/batch should control the simulator at a time. The baseline
uses monotonic wall time, including child startup/JIT compilation, so runs are
repeatable configurations, not deterministic trajectories. Browser polling
has no effect on metric collection. Stop run cancels a measurement and pauses
motion; the existing Realtime button only stops browser polling. Editing a
configuration invalidates an active measurement (`config_changed`).

Arrival means XY point-goal error <= 0.35 m. This is an evaluation threshold,
not an existing controller arrival signal. Collision uses the simulator's
existing geometric collision latch and terminates a run. Imported voxel-map
collisions are not measured by that latch. Stuck means at least 3 s of recent
history (up to 8 s), <0.1 m movement and <15 degrees turning; count transitions
into that state. Stuck does not terminate the run. Timeouts default to 120 s.

Finished records are saved under `tinynav_temp/navigation_lab/<run-id>.json`.
Export JSON includes the initial config, settings, definitions and pose samples.
Recent arrival rates use at most 100 records, grouped by exact config hash and
timeout. Cancelled/changed runs are excluded. Batch records and summary are
saved under the selected `--output` folder. No shortest-path/SPL claim is made.

## WorldState

The default `observed` source reconstructs sampled synthetic depth rays with
camera intrinsics and extrinsics. Only positive measured depth carves free cells;
zero/no-return rays remain unknown. Obstacles are endpoint hits in the same
camera-relative vertical band used by planning, with ground excluded. World XY
cells at 0.1 m retain observations until config reset/change. The current adapter
projects the observed body band into 2D; it is not a complete 3D visibility or
safe-corridor model. Eight center rays expose known clear distance and where
observation stops (`blocked`, `unknown`, `range_limit`). They are not swept
footprint clearance and must not be used directly as a motion safety guarantee.

Goal bearing and distance are robot-relative. Goal coordinates are explicitly
user supplied, not inferred detections. Recent movement, turning, goal progress
and movement pattern use the pose history. No object names, doorway labels or
scene object coordinates enter observed mode. It does not infer semantic labels.

`full_scene` is a debugging source that queries synthetic box geometry, ignoring
visibility. Imported maps are explicitly unknown in that mode. The goal-ray
query is capped at 5 m; `range_limit` does not imply the rest of the route is free.
Switching source does not alter planning or measurements.

API: `/api/baseline/scenarios`, `/api/baseline/start`, `/api/baseline/stop`,
`/api/baseline/status`, `/api/baseline/export`, `/api/baseline/results`,
and POST `/api/world-state/mode?mode=observed|full_scene`.

Validation: `uv run python -m unittest discover -s tests -p test_navigation_lab.py -v`.

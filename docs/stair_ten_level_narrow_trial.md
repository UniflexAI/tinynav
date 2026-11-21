# Ten-level, 1.2 m flight-width simulation

Updated: 2026-11-21. Branch: `xiaole/stair-floors`.

## Scene and command semantics

The scene has ten full-floor platforms: starting level 10 through level 1, with
nine full-floor height differences. Top-to-bottom traversal is therefore a
relative `DOWN 9` task, or eighteen flights. `--floors 10` would descend ten
full-floor differences and requires eleven platform levels.

Geometry in this case:

- Clear width of each flight: 1.2 m, measured between side boundaries.
- Turning-platform depth along the incoming flight: 1.75 m.
- Width across both flights and the central opening: 2.54 m (1.2 + 0.14 + 1.2).
- Eight modeled treads per flight, 0.30 m run and 0.16 m rise; the following
  platform is one further rise below, totaling 1.44 m per flight.
- Eighteen flights, seventeen intermediate turning platforms, target platform
  at z=-25.92 m relative to the top floor.

This changes flight width; it does not set the turning-platform depth to 1.2 m.
No random doors are enabled in this regression. The same measured front camera,
existing target/planner/controller, OCC behavior and shaft assertions are retained.
The geometry parameters are simulation inputs, not ground-truth inputs to navigation.

## Reproduce

Prepare the runtime/container as described in
[the deployment handoff](stair_floors_ubuntu_x64_deployment.md#11-optional-isolated-mujoco-regression-on-x64).
Inside that isolated simulation container, with ROS sourced and simulation
requirements on PYTHONPATH:

```bash
python -u tool/stair_mode_sim.py \
  --sim-root /simrepo \
  --levels 10 --floors 9 \
  --flight-width 1.2 --landing-depth 1.75 \
  --capture /simwork/output/ten-level-narrow.npz
```

A shorter task on the same building uses `--levels 10 --floors 2`. It must stop
at the second relative floor without requiring the rest of the building to end.
Omitting the new layout arguments preserves the previous one/two-floor cases.
The maximum simulation task count now matches the App's existing 50-floor limit;
this is argument support, not validation of all those trajectories.

The harness budgets at least 90 simulated seconds per requested flight, with a
minimum total budget of 180 seconds. It does not remove any count, contact,
fall, shaft or stopping checks to accommodate longer tasks.

## Historical first result: NOT PASSED (before the fix)

Both the short task and the full DOWN 9 request reproduced the same failure
on the first flight, before any landing count. At about 21.4 seconds after task
start, the robot was drifting toward the railing. The independent 1.0 x 0.5 m
rectangular-footprint check reached zero clearance from the central opening:

```text
SCENE levels=10 flights=18 flight_width=1.2 landing_depth=1.75 task_floors=9
...
down ok ... landings=0/18 floors=0/9 recovery=0
AssertionError: ('shaft', array([1.9095495, 0.32539567, -0.74217504]))
```

The assertion terminates the simulation. It is not evidence that the deployed
planner detected the opening and commanded a protective stop. At that revision navigation had not passed continuous traversal in this narrower scene; do not
use this scene as a successful ten-level demonstration or a physical deployment
qualification. No navigation thresholds or footprint margins were reduced.

The capture has `arrived=False` and `min_clearance=0`, plus actual local BEV,
planner path/target, controller commands and layout metadata. File existence alone
is not a pass. The last captured frame can precede the failing physics substep;
`min_clearance` records the independent substep check as well.

Geometry and existing stair-mode tests pass 31/31. The existing default DOWN 2
integration regression still passes, with the same four count events, final stop
and 0.4954 m minimum shaft clearance. This isolates the new narrow-layout failure
without claiming that the planner has been fixed for it.

The investigation then inspected the entry alignment and cross-flight target
position, body-footprint support at the railing edge, and whether tread-axis
alignment overrides useful lateral centering. Keep physical footprint and opening
checks intact. Repeat the narrow first-flight case before attempting eighteen
flights, and record any actual navigation changes separately from scene changes.

## Final x64 regression (2026-11-21)

The first-flight bias came from aligning to measured tread planes only after
`on_flight` and `flight_dir` were established. Before pitching onto the flight,
the open-space target could already steer toward the railing. Tread-plane
alignment now also runs for an `ok` entry target, using camera heading until a
travel direction is available. The observed/reachable and plane-fit constraints
remain. Tests cover UP/DOWN and entry headings of -11/+11 degrees.

After this fix the original 1.75 m platform-depth case passed DOWN 9 and UP 9.
Reducing platform depth to 1.2 m then exposed a separate terminal issue: both
runs counted all eighteen flights but latched arrival before fully entering the
last platform (x about -0.43 m). The retained x < -0.5 m final-position assertion
correctly failed both runs.

Final count now enters `finishing`: move along the last flight direction onto
observed, reachable flat support with obstacle clearance and forward bearing;
do not start another U-turn. Arrival requires 0.4 m progress beyond the last
pitched pose, level attitude and nearby measured flat support. No candidate means
stop/wait, not success. Depth/odometry freshness checks remain active. No world
platform dimensions or final-position assertion coordinates enter target generation.
Failure captures now distinguish node arrival from independent validation success.

| Platform depth | Direction | Flights | Arrival, simulated seconds | Minimum shaft clearance | 1.5 s hold drift |
| --- | --- | --- | --- | --- | --- |
| 1.75 m | DOWN 9 | 18 | PASS | 0.148643 m | hold passed |
| 1.75 m | UP 9 | 18 | PASS | 0.205836 m | hold passed |
| 1.2 m | DOWN 9 | 18 | 382.5 | 0.129389 m | 6.08 mm |
| 1.2 m | UP 9 | 18 | 366.5 | 0.215729 m | 13.69 mm |

All four final-version processes exited 0. The compact captures independently
confirm arrived=True, eighteen ordered landing events, direction/dimensions and
zero commands after arrival. Count-height tolerances (0.18 m), final height
(0.12 m), final position, fall, contact, independent 1.0 x 0.5 m rectangular shaft
footprint and 1.5-second stop checks were retained. No assertion was loosened.

The compact platform is **1.2 m deep by 2.54 m across the two flights**, not a
1.2 m square. UP is a signed ascending stack with relative target height +25.92 m;
DOWN is -25.92 m. The flight-width and depth arguments describe simulation only.

The focused suite passes **41 tests**, including geometry for both signs and both
depths, entry alignment, final support advance, missing support and facing-away
stop. The new final-support tests fail against the old immediate-arrival callback.
This is a fixed-scene regression, not a stochastic success-rate measurement.

Use [the development guide](stair_floors_development_simulation.md) to reproduce.
Existing 3090 evidence is under
`/home/dm/tinynav-stair-dev/experiments/landing-1p2/`: `verified-down.log`,
`verified-up.log`, `verified-original-down.log`, `verified-original-up.log` and
`result.json`. Compact NPZ/MP4 files are
`/home/dm/tinynav-stair-sim/output/compact-verified-{down,up}-nine.{npz,mp4}`.
Large captures/videos and generated GPU engines are deliberately not stored in Git.

Machine-readable metrics and validated source hashes: [result JSON](validation/stair_floors_2026_11_21.json).

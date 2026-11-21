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

## First result: NOT PASSED

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
planner detected the opening and commanded a protective stop. The current
navigation has not passed continuous traversal in this narrower scene; do not
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

Next navigation work should inspect the entry alignment and cross-flight target
position, body-footprint support at the railing edge, and whether tread-axis
alignment overrides useful lateral centering. Keep physical footprint and opening
checks intact. Repeat the narrow first-flight case before attempting eighteen
flights, and record any actual navigation changes separately from scene changes.

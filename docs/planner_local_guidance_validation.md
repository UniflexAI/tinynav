# Planner local guidance validation

## Scope

Branch: `xiaole/planner-local-guidance`, based on `main` at `f43fc61`.

This branch adds obstacle-aware local waypoint selection to normal planning. It has no independent stall/hold timer, recovery state machine, scan, retreat sequence, attempt memory, or near-goal finishing behavior. Only measured obstacle endpoints are retained for route selection. The existing live ESDF footprint scorer checks candidate motion; unknown space is handled as in the normal planner, not as observed-clear recovery coverage.

Normal trajectory generation, scoring, and reverse hysteresis from current main remain intact. A short local bypass uses finer trajectories capped at 0.15 m/s and the previously tested local clearance scoring. Large detours are rejected. ROS topics, controllers, Websim UI, and scenario definitions are unchanged.

## Method

All seven presets from the existing recovery Websim were tested, plus the previously edited S-bend. Six presets exist in main; the dead-end geometry was supplied externally for validation. The edited S-bend changes upper_wall_entry center XY to [1.14, 0.7] and right_deflector center XY to [2.72, 1.04].

Tests ran in isolated ROS domains on spark, using GO2, a 160x100 depth camera, fx=80, fy=50, and camera height 0.45 m. Each scene started from its configured initial pose. Arrival means XY goal distance <=0.35 m; failure means a geometric collision or a 120 s wall-clock timeout. Batches ran concurrently in separate simulator processes; timings are not deterministic benchmarks. This is one trial per case, not a statistical success-rate estimate.

## Results

| Scene | Outcome | Elapsed (s) | Final goal distance (m) | Collision |
|---|---|---:|---:|---|
| l_turn | arrived | 72.46 | 0.330 | No |
| straight | timeout | 120.70 | 0.391 | No |
| s_bend | arrived | 48.31 | 0.320 | No |
| s_bend_edited | arrived | 40.25 | 0.314 | No |
| narrow_gate | timeout | 120.70 | 0.544 | No |
| open_target | arrived | 38.21 | 0.324 | No |
| back_target | arrived | 10.05 | 0.319 | No |
| dead_end | timeout | 120.72 | 2.774 | No |

Five cases arrived and three timed out; no collisions were recorded. Dead-end recovery is explicitly out of scope. Straight and narrow-gate trials reached the goal vicinity but did not meet the strict arrival threshold.

## Main diagnostics

Unmodified main was tested separately on Straight and Narrow gate for 40 s to check the near-goal stalls:

- straight: stopped 0.391 m from the goal, no collision.
- narrow_gate: stopped 0.520 m from the goal, no collision.

These diagnostics reproduce the same category of near-goal behavior on main; they do not establish equivalence across all scenes. This branch deliberately does not restore recovery's near-goal finishing policy or change the arrival threshold to count these as successes.

## Automated checks

- 12 targeted local-routing, observation, cache, and navigation-reset tests passed.
- 6 existing main planning/heading checks passed.
- Python compilation and `git diff --check` passed.

Physical robot execution has not been validated. Raw per-second trajectories and configurations are retained with the local validation artifacts; they are not included in this small code change.

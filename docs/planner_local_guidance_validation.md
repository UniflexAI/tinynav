# Planner local guidance validation

## Scope

Branch: `xiaole/planner-local-guidance`, based on `main` at `f43fc61`.

This branch adds obstacle-aware local waypoint selection to normal planning. It has no independent stall/hold timer, recovery state machine, scan, retreat sequence, attempt memory, or near-goal finishing behavior. Only measured obstacle endpoints are retained for route selection. The existing live ESDF footprint scorer checks candidate motion; unknown space is handled as in the normal planner, not as observed-clear recovery coverage.

Normal scoring and reverse hysteresis from current main remain intact. Inside 1 m of the goal, the normal candidate set also includes a slower lattice capped at 0.15 m/s, evaluated with the same original cost and footprint checks. A short local bypass uses finer trajectories capped at 0.15 m/s and the previously tested local clearance scoring. Large detours are rejected. ROS topics, controllers, Websim UI, and scenario definitions are unchanged.

Local route selection, observed-obstacle storage, and cache lifecycle are implemented directly in `planning_node.py`; there is no separate planner or guide object.

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

These diagnostics reproduce the same category of near-goal behavior on main; they do not establish equivalence across all scenes. The initial branch did not address this issue. A subsequent planner-only fix adds slow near-goal candidates without changing the arrival threshold or restoring recovery actions. The original results above describe the initial implementation.

## Automated checks

- 12 targeted local-routing, observation, cache, and navigation-reset tests passed.
- 6 existing main planning/heading checks passed.
- Python compilation and `git diff --check` passed.

Physical robot execution has not been validated. Raw per-second trajectories and configurations are retained with the local validation artifacts; they are not included in this small code change.

## Follow-up: inline routing and slow goal approach

Local routing now lives directly in PlanningNode, without a separate guide module or object. The 12 targeted tests passed again after this change. Routing and cache methods were checked for structural equivalence to their previous implementation.

After adding slow candidates inside 1 m of the goal, isolated Websim retests reported:

| Scene | Outcome | Elapsed (s) | Final goal distance (m) | Collision |
|---|---|---:|---:|---|
| straight | arrived | 13.07 | 0.294 | No |
| narrow_gate | timeout | 40.21 | 0.539 | No |

Narrow gate still oscillates near the goal: the recorded commands include a reverse command as it approaches the end wall. This remains unresolved. The other scenes were not rerun after the slow-candidate change; their earlier results must not be treated as validation of the final revision. Physical robot validation remains pending.

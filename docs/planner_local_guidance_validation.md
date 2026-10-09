# Unified planner validation

## Implementation

Branch: `xiaole/planner-local-guidance`, based on main at `f43fc61`.

All runtime changes are in `tinynav/core/planning_node.py`. The planner selects a current target (the task goal on a clear route, otherwise a short local waypoint), generates one trajectory library, and selects motion with one cost function. The final task goal does not change.

- Forward sampling always includes low speeds. Its upper limit decreases with the distance to the current target; reverse candidates remain in the same library.
- All candidates use the same distance, heading, clearance, and velocity-change weights. There is no local mode, candidate-group switch, or dedicated guide trajectory library.
- The previous front-distance reverse gate and its hysteresis state are removed. Forward and reverse candidates compete under the same cost and footprint collision checks.
- Local routing retains measured obstacle endpoints, prefers clearance, and rejects long detours while searching. Targets outside the local map are treated as directions for useful local progress, not clipped terminal goals.
- Target changes, POI changes, pause, and navigation changes invalidate route memory. Live occupancy decay, obstacle inflation, footprint checks, safety radius, ROS topics, controllers, Websim UI, and scenes are unchanged.

This revision removes 47 runtime lines relative to the preceding shared-scoring implementation. It does not include a model, full-turn scan, retreat sequence, or a recovery state machine. Retained obstacles influence routing; candidate motion is still checked against the live ESDF. Unknown space is treated as in the existing planner.

## Method and results

Tests ran on Spark in isolated ROS domains, using GO2 and a 160x100 depth camera (fx=80, fy=50, mount height 0.45 m). Arrival means XY goal distance <=0.35 m; the timeout is 120 s. Each test resets its scene. These timings are simulation observations, not deterministic performance benchmarks or statistical success rates.

| Scene | Outcome | Elapsed (s) | Goal distance (m) | Collision |
|---|---|---:|---:|---|
| l_turn | arrived | 48.26 | 0.340 | No |
| straight | arrived | 11.07 | 0.269 | No |
| s_bend | arrived | 47.29 | 0.345 | No |
| s_bend_edited | arrived | 39.20 | 0.314 | No |
| narrow_gate | arrived | 21.13 | 0.308 | No |
| open_target | arrived | 41.22 | 0.313 | No |
| back_target | arrived | 4.02 | 0.300 | No |
| dead_end | timeout | 120.71 | 2.940 | No |

L-turn also arrived in a separate uninterrupted trial in 55.31 s at 0.328 m, without collision. Narrow gate repeated in 22.12 s at 0.307 m; during a subsequent 20 s observation it remained inside the arrival threshold, ending at 0.179 m with zero velocity and no collision.

An initial unified trial stalled in L-turn. Removing mode switching exposed route selection that could hug the entry wall or discard all progress after rejecting one long route. Clearance preference and rejection during search corrected this regression; the table reports the final revision. Dead-end escape remains unsupported and physical robot execution has not been validated.

## Automated checks

- 15 targeted routing, observation, navigation-reset, corridor-clearance, and long-route rejection tests passed.
- 6 existing heading/planning checks passed.
- Python compilation and `git diff --check` passed.

Raw scene configurations and per-second observations are preserved on Spark in `/tmp/local-guidance-unified-validation/`; they are not added to this small PR.

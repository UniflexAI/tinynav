# Planner maintenance fixes

Based on main fc17408, branch xiaole/planner-fixes.

## Scope

- Increase the 0.05 m grid height from 10 to 40 cells (2 m total).
- Validate and forward complete WebSim robot settings to planner and simulator controller. Show values confirmed by the planner.
- Generate trajectories from t=0 to t=3.0 s and timestamp each published pose accordingly. Retain the original 106 candidates and control interval.
- Make the original per-update occupancy multiplier configurable, default 0.99. No elapsed-time conversion is applied.

Goal distance and velocity estimation remain unchanged from main. No two-stage candidates are included.

## Occupancy decay factor

WebSim exposes `planner.occupancy_decay_factor` as **Obstacle decay factor**. Each occupancy update performs `occupancy_grid *= factor`, adds current observation evidence, and clips values as before. Default 0.99 exactly retains the original update rule. Larger factors retain old evidence longer. Valid range is [0, 1]: 1 disables decay; 0 discards old evidence before adding the current observation. Actual decay in seconds depends on update frequency, as in main.

Edit the input and start Realtime to apply. Changing the factor resets the scene and restarts the planner. The server passes TINYNAV_OCCUPANCY_DECAY_FACTOR to the child. `/planning/occupancy_decay_factor` reports the actual loaded value, also displayed in WebSim. `/api/effective-robot` exposes requested and confirmed factors separately. Non-finite, out-of-range and boolean values are rejected before replacing current config. The previous seconds-based setting is removed.

A standalone App does not yet expose this setting. This change provides the WebSim frontend/backend path; production startup may use the environment variable.

## Validation and limitations

Five focused tests pass: trajectory origin/endpoints, published timestamps, per-update decay at factors 0, 0.9, 0.99 and 1, and robot configuration validation. Python/JavaScript syntax and whitespace checks pass. API integration confirms factor-only changes from 0.95 to 0.99 with reset=false; negative, greater-than-one and boolean values are rejected and defaults are restored.

Earlier navigation runs preceded the final scope adjustments and are not acceptance results for this revision. GO2 hardware and Nano performance remain untested. Corner deadlock remains unresolved.

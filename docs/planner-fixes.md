# Planner maintenance fixes

Based on main fc17408, branch xiaole/planner-fixes.

## Scope

- Increase the 0.05 m grid height from 10 to 40 cells (2 m total).
- Validate and forward complete WebSim robot settings to planner and simulator controller. Show settings confirmed by the planner.
- Generate trajectories from t=0 to t=3.0 s; stamp published poses at their respective future times. Retain the original 106 candidates and control interval.
- Decay unobserved occupancy by monotonic elapsed time rather than once per frame. Measurement evidence still accumulates per observation.

Goal distance and velocity estimation are unchanged from main, as requested. No two-stage candidates are included.

## Decay configuration

WebSim exposes `planner.occupancy_decay_tau_s` as **Obstacle decay time constant (s)**. Default: 10 seconds. Unobserved evidence is multiplied by exp(-elapsed/tau); its half-life is tau*ln(2), approximately 6.93 seconds at the default. Larger values retain old evidence longer. This is a decay constant, not a fixed obstacle expiration time.

Edit the input and start Realtime to apply. Changing this setting resets the scene and restarts the planner. The server passes TINYNAV_OCCUPANCY_DECAY_TAU_S to the child. `/planning/occupancy_decay_tau_s` reports the actual loaded value, also displayed in WebSim. `/api/effective-robot` returns requested and confirmed values separately. Non-finite, zero, negative and boolean settings are rejected before replacing the current config.

A standalone App does not yet expose this setting. This change provides the WebSim frontend and backend path; production startup may use the environment variable.

## Validation and limitations

Five focused tests pass, covering trajectory origin/endpoints, published timestamps, elapsed-time decay and robot configuration validation. Python/JavaScript syntax and whitespace checks pass. API integration confirms changes from 18 seconds to 6 seconds, including parameter-only changes with reset=false; zero, negative and boolean values are rejected and defaults are restored.

Earlier-revision navigation checks reached an unobstructed 2 m target but still reproduced L/S deadlock. Those runs preceded reverting goal-distance and velocity-estimation changes and are not acceptance results for this revision. GO2 hardware and Nano performance remain untested; corner deadlock remains unresolved.

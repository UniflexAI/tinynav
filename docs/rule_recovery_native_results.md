# Native controller recovery validation (2026-10-06)

The web simulator used the actual robot `cmd_vel_control.py` controller, with
rule recovery enabled/disabled, and the unchanged original planner. Both runs
used the same dead-end scene, initial pose, mission goal [4, 0, 0], configuration
hash `d39742a19c2f711a`, and 180 s limit. The disabled baseline was collected
before the final finishing rule; that rule has no effect when disabled.

| Mode | Result | Wall time | Final goal distance | Collisions |
| --- | --- | --- | --- | --- |
| Rule on | Arrived | 134.007 s | 0.341 m | 0 |
| Rule off | Timeout | 180.111 s | 3.107 m | 0 |

On run: `35b688f7e28f40fcb7d1b7222e0ea559`. Off run: `4d726a91665741b7901e235f9432ecc1`. Raw navigation records persist in
`tinynav_temp/navigation_lab/<run-id>.json`; paired result is saved in
`tinynav_temp/rule_recovery_native_final.json`.

The rule scanned, selected an observed long detour, executed it, and restored
normal path following. Near the goal the original controller stalled outside
the 0.35 m arrival threshold; the observed bounded finishing rule completed
arrival. No model, hidden waypoint or mission-goal replacement was used.

## Integration findings retained rather than counted as successes

- Initial ROS velocity construction used integer zeros and was rejected;
  converted fields explicitly to floats and added adapter tests.
- Reducing speed originally reduced the fixed-duration probe distance;
  changed the stage to preserve its distance.
- The simulator-only controller and robot controller handle reverse turns
  differently. Short recovery returned to the trap; added a longer fully
  observed detour and preferred longer clear sides when attempt counts tie.
- A 120 s run reached 0.449 m but timed out. A subsequent run stalled near the
  goal. These are not arrivals. Added one 15 s finishing attempt within 0.65 m,
  requiring observed clear swept footprints and fresh sensors.

## Verification and scope

41 targeted tests passed, covering measured observations, candidate footprint
rejection, attempts and scan budget, pose/target freshness, pause/inactivity,
ROS float fields, single-controller command arbitration, small localization
jitter, minimum executable final scan turns and bounded goal finishing.
The page exposes five scenes and only rule on/off modes, defaults to 180 s,
and has no model observer controls or JavaScript errors in browser verification.

This is one final dead-end pair in static simulation, not a five-scene repeated
success-rate benchmark or a physical-robot test. The other four scenes remain
available for testing. Historical model experiments were removed from this
branch's source tree and remain recoverable in the observer branch/archives.

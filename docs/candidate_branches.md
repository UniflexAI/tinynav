# Diverse candidate / isolated branch evaluation

The model now receives the planner selection plus the cheapest finite candidate
from forward-straight, forward-left, forward-right, reverse, turn-left and turn-right.
Duplicates are removed. Each direction reports candidate counts and collision rejection
counts, including unavailable directions. Collision-rejected candidates are not offered.
Reverse-gate penalty candidates can be offered for isolated evaluation: a policy penalty
is not a collision rejection. They are never applied to ROS control.

Camera-down angular parameters have the opposite sign to control yaw; report now also
contains control_linear_mps from the trajectory's first published segment (poses 0 and
10 over 1s), matching simulator_control's forward projection, and control_yaw_radps.

POST /api/decision/compare?decision_id=<id> uses the frozen full scene in the saved
analysis context and the planner's input pose/yaw. It compares all supplied finite
candidates in separate in-memory copies: 3 seconds of constant first-segment command,
50ms Euler steps with the same XY-footprint / box collision test as websim. Stops on
collision. Unknown ground truth is used only by evaluator, never model input. Imported
voxel maps are explicitly unsupported. No live pose, planner, target or cmd_vel is changed.
The model branch is null when its output is uncertain/request_new_candidates; those
outputs are not counted as baseline-equivalent successes. Wrong/stale decision IDs fail.

The Decision page includes an isolated branch comparison button. Results are stored in
tinynav_temp/candidate_branches/<decision-id>.json. Reproduce repeated snapshots:
python3 scripts/validate_candidate_branches.py --output <folder> --repeats 2
Each test runs the current planner for 12s, captures a decision snapshot and pauses;
both branches start from the same report pose. The offline evaluation is not closed-loop
replanning and cannot establish arrival rates, longer-term escape or control safety.

## 2026-10-03 actual-control-command results

Raw data: tinynav_temp/branch_validation_control_v2_20261003.
Three scenes, two snapshots each. 6 completed decisions, 5 paired branches, 1 uncertain.
Four improve short-term progress by >5cm; one keeps baseline. Zero model collisions in
these 5 branches. Selected alternative probabilities only 0.210–0.311.

| Scene / repeat | Model choice | Extra progress over planner (m) |
| --- | --- | --- |
| open_target / 1 | uncertain | not evaluated |
| open_target / 2 | candidate_18 | +0.3009 |
| l_turn / 1 | candidate_26 | +0.3073 |
| l_turn / 2 | keep_current | 0 |
| dead_end / 1 | candidate_19 | +0.4107 |
| dead_end / 2 | candidate_19 | +0.4107 |

An earlier nominal-velocity trial produced 3/6 short-term improvements; that trial used
lattice velocities without the controller's forward projection and is superseded by
these results. Input formatting and actual snapshot poses also changed between trials;
this is not a controlled attribution experiment. Low probabilities and differing
repeated-scene choices warrant further testing. No model feedback to live planning added.

Tests cover direction/sign diversity, penalized-but-finite alternatives, collision
rejection, isolated state, synthetic geometry collision, actual-command override and
unsupported map handling. Existing observer lifecycle and option validation pass.

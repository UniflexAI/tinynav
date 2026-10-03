# Planning-report observer

On spark, Planning Lab now sets TINYNAV_PLAN_REPORT=1 for its child nodes.
The planner publishes a JSON String on /planning/report for each planning cycle
with camera intrinsics available. Outside the websim launcher, reports default off.
Reports contain sequence/process ID, input timestamp, target/robot XY, speed limit,
front clearance, candidate velocities/endpoints/costs/obstacle and reverse penalties,
collision rejection reasons/counts, selected trajectory and planning duration.
Nonfinite collision costs serialize as null. Reports stay in ROS/latest memory;
only snapshots submitted for analysis are written to decision_observer JSON logs.
GET /api/planning/report returns the latest full report and reception age.
Configuration reset clears the cached report and rejects messages from older input
stamps; analysis rejects missing reports or reports received more than 3s ago.

Analysis submits WorldState plus the top five finite candidates and two collision
examples, not the full candidate table. Full report is preserved in record.context.
The motion pattern label is removed from model input. Four typed questions ask
for action, stuck probability, planning_issue, and alternative (keep current,
request new candidates, uncertain, or one supplied finite ungated candidate).
Response validation rejects options outside the supplied list and invalid probabilities.
Different questions can disagree; their outputs are displayed unchanged for diagnosis.
Candidates outside the top five are not analyzed as alternatives in this version.

The Decision page supports manual analysis or optional browser-driven periodic
analysis while the page remains open. Minimum submission interval 2s; one request
at a time, so longer inference automatically lowers the rate. Paused navigation
stops automatic submission. Manual analysis can inspect fresh paused reports.
No model output is connected to path publication, targets or cmd_vel.

Near-goal regular velocity sampling now caps its maximum at remaining XY distance
/ 3s (the existing trajectory horizon). Previously the fixed lattice selected zero
velocity around 0.50m. The baseline arrival radius remains 0.35m.
This changes the planner's near-goal behavior, including non-websim users of the
planner; physical robot deployment has not been tested or performed.

## Verification on spark

- Empty scene: 2/2 arrived, elapsed 11.1s and 9.1s; radius unchanged at 0.35m.
- Four-question empty-scene analysis succeeded in 2447.9ms; action continue,
  planning_issue none, keep_current probability 0.785.
- Dead-end report: 106 candidates, 67 collision rejections; selected zero velocity.
  Analysis succeeded in 2699.3ms, planning_issue collision_blocked (0.434),
  action replan (0.350), alternative keep_current (0.510). This shows disagreement
  and limited confidence; it does not establish useful escape behavior.
- Missing report returned HTTP 409; browser manual and automatic analysis verified.
- Observer lifecycle/probability tests, dynamic candidate filtering/invalid-option
  tests and NavigationLab tests passed.
- Full records in tinynav_temp/plan_report_validation and decision_observer.

Model-in-loop control and improvement in arrival rate remain untested.

## Follow-up
Candidate selection and branch evaluation are extended in candidate_branches.md.
The current model input includes direction representatives, and finite reverse-gate
penalty candidates are available only for isolated evaluation.

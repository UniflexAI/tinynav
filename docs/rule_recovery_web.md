# Rule-assisted recovery web experiment

New explicit `TINYNAV_EXPERIMENT_MODE=rules` mode uses the existing simulator and planner. Experiments require ROS localhost isolation and a loopback HTTP listener; this restriction is unchanged. The normal 8766 service remains off/basic.

## Policy

A fresh report (<0.5 s) and a measured window of at least 6 s with <0.1 m movement trigger a scan, unless another override or recovery is active. Scan translation is zero, angular speed is bounded by both 0.6 rad/s and the robot limit. Abort on collision, >2 cm translation, stale report, or >25 s duration. Wait for a new report received after scan completion before selecting any action.

Every candidate's complete sampled footprint must have zero unknown and blocked cells. Prefer a unique fully observed pivot with predicted goal progress >0.1 m. Otherwise keep the longest fully observed retreat on each side (at most two), selecting the less previously attempted ID with a stable ID tie break. If no retreat remains, select the fully observed progressive pivot with greatest predicted progress. None of these rules is a model preference.

The existing recovery executor retains per-stage collision and report-freshness checks. The original planner takes over after recovery; mission arrival is always measured against the original target. Maximum three scans per run, including scans that find no executable candidate. Model workers are not invoked in rules mode. No planner code changed.

## Web page

`/decision` shows rule mode, stage, scan rotation, selected/excluded IDs, exact unknown/blocked footprint cell counts, execution stages, previous outcomes, and original-goal arrival metrics. Start dead-end or L-turn runs, stop, or download current experiment JSON. Completed rules runs also persist experiment state, events, and attempts alongside navigation samples under `tinynav_temp/navigation_lab`.

## Service

```bash
docker exec -d -e ROS_DOMAIN_ID=216 -e ROS_LOCALHOST_ONLY=1 \
  -e TINYNAV_EXPERIMENT_MODE=rules -e TINYNAV_WEB_HOST=127.0.0.1 \
  -e TINYNAV_WEB_PORT=8774 tinynav-dev bash -lc \
  'cd /tinynav && bash scripts/run_ros_planning_web.sh > /tmp/rules-web-8774.log 2>&1'
```

One isolated service is already running. Do not start another copy on the same domain/port. Local browser access uses an SSH tunnel from Mac localhost:18774 to spark localhost:8774. Closing that tunnel removes local access without changing the main service.

## Validation and limits

9 policy unit tests, 3 retreat-selection tests, and 8 existing observation/screening regression checks pass. Python compiles and browser JavaScript parses. Actual browser state confirms live stage, candidates, execution, and arrival presentation.

Full-start live HTTP pair with the final action-selection policy: L-turn arrived in 88.390 s after three scans; dead-end arrived in 48.539 s after one scan. Both had zero collisions and no model calls. These are individual synthetic runs, not a general success-rate estimate. Early retreat-only and no-retreat-refusal variants failed L-turn and were replaced; results are retained locally for comparison. Report/coverage persistence was then checked in a final restarted service run.

Model order sensitivity and independent preference quality remain separate unresolved issues. This explicit rule-assisted mode does not claim to fix them or demonstrate model-controlled navigation.

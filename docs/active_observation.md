# Sensor-only active observation experiment

This script reuses RosPlanningSimNode.tick, command integration, robot collision rollback, camera pose, render_depth and NavigationLab.observe_depth from the existing websim. It instantiates the node in local-only ROS domain 218, invokes its ticks explicitly, and never starts an HTTP server, planner/control child process or model worker. The ordinary websim service is unchanged. This is a websim-code observation replay, not a live navigation/escape trial.

Seven previously measured snapshots are restored at their captured pose with the corresponding recorded synthetic scene. Their bounded observed-cell audit is restored with saved observation ages preserved relative to the new replay clock. Cells remain unknown if absent. Scene object geometry is used only by the existing depth renderer and collision guard, never directly copied into observed occupancy or model evidence. Imported maps are rejected.

Two separate branches start from the identical snapshot: zero-linear-velocity 360-degree rotation at up to 0.6rad/s, alternating rotation direction between snapshots, and stationary forward observation for about eleven seconds. Scan completion requires at least 359.5 degrees of measured rotation; collision or a 45-second deadline aborts. Zero translation is checked, actual heading error is logged, and the original tick collision guard remains active. The scan lasts roughly 10.7 seconds; stationary controls are slightly longer, roughly 11 seconds. They are duration-comparable rather than exactly synchronized trials.

The initial candidate command sequences are held fixed for before/after audits. Their sampled conservative footprint cells are evaluated at the original anchor pose/yaw, with exact blocked and unknown counts. The actual scan returns within 0.35 degrees of that anchor heading; no teleport is used to close the scan. A count of zero is required to mark a stage fully historically observed clear; rounded fractions cannot conceal a remaining unknown cell. After observation, the existing proposal generator is separately rerun to record which proposals survive newly measured blocking. Historical cell labels still do not prove current footprint safety or escape success.

Raw depth arrays from several scan orientations, actual motion frames, before/after cell labels/timestamps and candidate stage counts are saved per branch. plot_active_observation.py renders measured occupancy and actual simulated depth only; the overlaid retreat line is an unexecuted fixed candidate, not robot travel or a known-safe path. White depth pixels denote no return and stay unknown. No full-scene oracle is drawn as observed space.

Run inside the existing Docker container:

```bash
docker exec -e ROS_DOMAIN_ID=218 -e ROS_LOCALHOST_ONLY=1 \
  -e TINYNAV_EXPERIMENT_MODE=baseline -e TINYNAV_WEB_HOST=127.0.0.1 \
  tinynav-dev bash -lc 'source /opt/ros/humble/setup.bash && cd /tinynav && \
  /opt/venv/bin/python3 scripts/run_active_observation.py \
  --snapshots tinynav_temp/topology_pairs_20261003 \
  --decisions tinynav_temp/decision_observer \
  --output tinynav_temp/active_observation_20261004'
```

The process destroys its ROS node and shuts down its context in finally. There is no surviving experiment service to stop. The model, ComfyUI and normal websim are not restarted or reconfigured. The planner and live execution code are not modified.

Snapshots share static synthetic scenes, so this is a controlled sensor-coverage comparison, not independent environment success statistics. No retreat, recovery, arrival or model usefulness assessment is performed. A scan must complete before requesting a new planning/decision snapshot; its roughly 10.7-second duration exceeds the existing decision TTL and old results must not be carried across it. The earlier local screener's two-candidate cap and tie abstention are unchanged; dead-end scans yielding eight observed candidates would still require an explicitly defined shortlist/selection phase.

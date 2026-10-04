# Observed retreat selection and live execution

This experiment keeps the planner unchanged and the mission goal intact. It runs only on localhost ROS domain 217, with no HTTP server and no model worker. The normal web service remains off/basic.

## Policy

1. Freeze translation and obtain a fresh real planning report.
2. Rotate through 360 degrees using the actual simulated camera/depth observation pipeline. Refuse an incomplete scan, collision, or anchor translation exceeding 2 cm.
3. Recompute proposals from measured cells. Exclude any retreat whose sampled footprint contains even one unknown or blocked cell. Keep the longest observed-clear existing retreat for each side, at most two.
4. In rule-assisted mode, prefer the less previously attempted strategy; break equal evidence by stable strategy ID. Record this as a rule, never as a model preference. The validation harness explicitly chooses a side to test both directions.
5. Require a planning report younger than 0.5 s when starting. Use the existing recovery executor, with its stage collision guards and report freshness checks. Restore original planner control after recovery.
6. Arrival is measured against the original mission goal, not recovery progress or a temporary target.

The candidate shortlist is a standalone auxiliary module. It has not replaced the normal model selector or been enabled on the main web service. A model-only deployment still needs reliable usefulness decisions; rule-assisted success does not demonstrate model decision quality. A future online mode should expose rule-assisted fallback explicitly and avoid reusing pre-scan advice, whose 8-second TTL expires during a scan.

## Reproduce

Inside the existing container, set PYTHONPATH before sourcing ROS (sourcing supplies the ROS Python paths):

```bash
cd /tinynav
export PYTHONPATH=/tinynav
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=217 ROS_LOCALHOST_ONLY=1
export TINYNAV_WEB_HOST=127.0.0.1 TINYNAV_EXPERIMENT_MODE=baseline
uv run python scripts/run_escape_validation.py --snapshot 2 --side left \
  --output tinynav_temp/escape_validation_20261004/snapshot-2-left-clean.json
uv run python -m unittest discover -s tests -p test_observed_retreat_selection.py
```

Do not run another experiment in domain 217 concurrently. A process lock prevents concurrent instances of this harness. The lock does not cover other scripts.

## Limits

The four frozen dead-end snapshots are from the same synthetic scene, not four independent environments. Sensor ray observation is real within the websim; collision guards still use the simulator's scene geometry, as before. Scene geometry is not supplied to selection. Successful arrivals do not prove general navigation, physical robot safety, or model reliability.

An exploratory observed-frontier temporary-target branch collided and was discarded. One overlapping run was excluded and repeated after all other domain-217 processes stopped. Only clean single-run records count toward the reported results.

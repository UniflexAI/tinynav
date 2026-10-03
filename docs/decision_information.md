# Decision information ablation

`TINYNAV_DECISION_INFORMATION=basic` preserves the prior model input. `rich` adds information derived only from measured occupancy and odometry. It does not change planner code, recovery proposal generation, model questions/criteria, candidate commands, guards, or execution duration.

Enhanced input includes a 9x9 robot-relative sampled observation grid at 0.4m spacing; up to four breadcrumbs covering the whole traversed route; robot-relative goal coordinates; and separate blocked/unknown/past-route overlap fractions for each recovery stage. Prior attempts and post-planner goal progress remain included. Past traversal never makes an unknown cell clear. Cells retain observations until reset but do not have per-cell age, so the input explicitly warns about stale clearance.

Enhanced input retains the current selected single-step trajectory and aggregate planner diagnostics; unused single-step alternatives and collision examples are omitted because only recovery sequences are offered. The full original planning report stays in logging context and is restored for basic replay. Questions and recovery candidates remain identical.

Repeated proposal observation notes are stored once in `recovery.shared_strategy_note`. `decision_information.basic` restores the original per-proposal fields and, with the source compact report, restores planning details and removes enhancements, permitting same-snapshot comparison with identical questions and offered strategies. Compact CSV strings have explicit column legends and fit the current 4096-token model slot better than verbose nested evidence. An initial verbose input overflowed the slot; that incomplete run is excluded and preserved separately under `information_context_overflow_20261003`. A partly completed batch that still overflowed on dead-end inputs is excluded under `information_partial_overflow_20261003`. An incomplete startup race is likewise excluded under `information_startup_race_20261003`; the runner now waits for each service's readiness.

Live smoke verification runs two repeats per scene/group, 120 seconds per run. This is eight runs, not a statistically strong replacement for the prior 20-run benchmark. The model group uses enhanced information; baseline is the unchanged planner. Historic basic-model results are not a paired live control because ROS scheduling differs.

A second ablation takes the first/last completed enhanced snapshots from each model run. Each immutable snapshot is sent once with basic input and once with enhanced input, alternating order. Questions and command stages are identical. No commands are executed during replay. Choice changes and inference latency can be attributed to the input presentation under this limited comparison; changes do not demonstrate a successful escape. The replay takes place after live trials to avoid adding model contention to control timing.

```sh
TINYNAV_DECISION_INFORMATION=rich bash scripts/start_closed_loop_services.sh
python3 scripts/run_closed_loop_experiment.py --repeats 2 --timeout 120 --output tinynav_temp/information_rich_20261003
docker exec tinynav-dev python3 /tinynav/scripts/stop_closed_loop_services.py
docker exec tinynav-dev bash -lc 'cd /tinynav && /opt/venv/bin/python3 -m scripts.replay_information_pairs --runs tinynav_temp/information_rich_20261003 --decisions tinynav_temp/decision_observer --output tinynav_temp/information_pairs_20261003'
```

The normal websim remains off/basic. `/static/closed-loop.html` provides a read-only selection for the enhanced-information experiment and previous experiments. Raw live records and paired model requests/responses are retained in the dated folders. Synthetic object coordinates are never added to model input; the existing simulation execution guard continues using simulator geometry, as before.

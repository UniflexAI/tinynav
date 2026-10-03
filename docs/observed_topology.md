# Observed route connectivity, information v2

Planner, recovery generation, questions/choices, collision guards and stage commands remain unchanged. `TINYNAV_DECISION_INFORMATION=topology` replaces the v1 small grid/breadcrumb presentation with bounded route connectivity and observation-time evidence. The basic/v1 input remains available for frozen comparisons.

NavigationLab records the monotonic observation time separately from each occupancy label. Clear rays do not refresh a previously retained blocked label. Occupied returns refresh blocked timestamps. Reset and spatial pruning clear matching timestamps. These timestamps do not alter occupancy labels or the planner's ROS depth input.

Connectivity uses four-neighbor clear center cells within six meters of the current pose. Blocked and unknown cells never bridge components. Component area and an endpoint's connection to any observed center cell along past motion are evidence about sampled center space, not a footprint-safe path. Missing start/goal/endpoint observations yield unknown, not a claim of blockage or reachability. Routes outside the bounded region may be missed.

The route summary includes the historical entry coordinate, actual traveled distance, conservative footprint coverage of the past route, the fraction recently measured clear, and observed goal-center connection. Recent means <=10 seconds old and is an experimental information label, not a safety guarantee. Missing timestamps are not recent; stale clear cells remain labeled historical clear rather than being silently reclassified or considered safe.

For each existing recovery sequence the input adds end-center connection to the past route, component area, and change in distance toward the historical entry. This helps distinguish retreat toward the entry from immediate progress toward the goal. Each stage separately reports recent clear footprint fraction. The historical entry is not asserted to be a currently safe escape waypoint.

A full bounded observation audit is saved in logging context, never sent to the model: cell coordinates/labels/timestamps and capture time. The unchanged v1 state is also saved in context. A serialization issue in the first short smoke run prevented results being recorded; that smoke is excluded. The corrected smoke validates completed model calls and logging before longer tests.

Validation: two 120-second repeats per scenario/group (eight runs), followed by basic/v1/v2 evaluation of identical measured snapshots with rotating order. The replay executes no commands and checks identical questions and stage commands. It measures input sensitivity rather than navigation success. It uses snapshots from the v2 group's visited states and is a small selected sample.

```sh
TINYNAV_DECISION_INFORMATION=topology bash scripts/start_closed_loop_services.sh
python3 scripts/run_closed_loop_experiment.py --repeats 2 --timeout 120 --output tinynav_temp/information_topology_20261003
docker exec tinynav-dev python3 /tinynav/scripts/stop_closed_loop_services.py
docker exec tinynav-dev bash -lc 'cd /tinynav && /opt/venv/bin/python3 -m scripts.replay_topology_pairs --runs tinynav_temp/information_topology_20261003 --decisions tinynav_temp/decision_observer --output tinynav_temp/topology_pairs_20261003'
```

The normal websim remains off/basic. Results can be selected at `/static/closed-loop.html`. These synthetic static-scene tests do not establish real-robot safety or clearance from old observations.

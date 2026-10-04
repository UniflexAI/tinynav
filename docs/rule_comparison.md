# Web rule-assisted on/off comparison

On the isolated rules service, each baseline start may specify `rules_enabled: true|false`. Omission keeps the existing enabled behavior. Other service modes reject this new field. Changing the selection applies only to a new run: all runs reset scene, pose, observations, recovery state, and planner/control children. The planner code and original mission goal do not change.

With assistance disabled, the rules worker only reports planner-only status: it does not scan, shortlist, call a model, or start a recovery. Mode is included in live status and saved experiment metadata. Completed comparison history survives service restarts via `/api/experiment/comparisons`, using the latest 40 navigation records with explicit on/off metadata.

The same-screen page offers on/off selection, a shared time limit (default 60 s), start/stop, a large actual-run mode banner, and recording view. Recording view hides raw evidence and model observer tools; uncheck it to inspect those details. Results pair only equal configuration hashes and time limits. The table shows the latest result in each mode, not an aggregate success rate; stopped runs are labeled as manually stopped.

Live validation, same dead-end configuration hash `b7642bc2a59c9d61`, same 60 s limit:

| Mode | Result | Elapsed | Collisions | Final distance |
|---|---|---:|---:|---:|
| Off: original planner | Timeout | 60.021 s | 0 | 3.104 m |
| On: rule-assisted | Arrived | 48.252 s | 0 | 0.345 m |

Disabled mode was polled throughout and verified to have zero assistance events, attempts, and overrides. The history API recovered both records after service restart. Browser state verifies the mode selector, recording switch, and paired result table. Python compiles, browser scripts parse, and the nine policy tests pass.

This is one synthetic pair. Wall time includes planner startup and scan time, and run timing is not deterministic. These results demonstrate rule-assisted behavior, not model/JEV decision quality. The main 8766 service remains off/basic.

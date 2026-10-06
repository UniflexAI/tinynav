# Rule-assisted recovery

Branch: xiaole/rule-assisted-recovery. This branch includes the measured recovery
policy and five-scene rule-on/rule-off Websim comparison, before model shadow
integration. Rule mode does not invoke the model worker or require an API key.
Historical observer experiments remain available but are not the rule policy.

## Run

Inside the ROS-enabled container, run:

```bash
bash scripts/run_rule_assisted_web.sh
```

Defaults: ROS domain 216, localhost-only ROS, loopback HTTP port 8774. Do not
start another instance on an occupied domain/port. Stop the existing isolated
service before replacing it. The primary service on 8766 is a separate instance.
Forward localhost:8774 over SSH and open /decision. Select a scene, choose rule
assistance on/off, and start. Each start resets pose, goal and observed cells.
Rule mode is opt-in; the regular launcher retains its existing default.

## Policy

At least six seconds of less than 0.1 m movement triggers an in-place scan.
Reports must be younger than 0.5 s. Scan aborts on collision, translation over
2 cm, stale reports or a 25 s timeout. After a complete scan, wait for a new
report. Candidate swept footprints must have zero unknown and blocked cells.
Prefer a unique progressive pivot; otherwise shortlist the longest fully
observed retreat per side and use attempt history and a fixed ID tie break.
If no retreat exists, use the most progressive observed pivot. No eligible
candidate means no recovery command. Keep existing execution collision and
freshness checks, at most three scans and an eight-second cooldown.

Recovery returns control to the unchanged original planner. The mission goal
never changes. This is a simulator integration, not a physical-robot deployment.

## Measured effects

| Scenario / comparison | Original planner | Rule assistance |
| --- | --- | --- |
| Dead end, same configuration, 60 s | Timeout, 3.10 m from goal | Arrived in 48.25 s, zero collisions |
| L turn, separate 120 s run | Not a matched pair | Arrived in 88.39 s, zero collisions |
| Rear goal, matched 60 s pair | Arrived in about 10.7 s | Arrived in about 10.7 s |

Four frozen dead-end recovery validations arrived with zero collisions in
38.63-39.02 s. These start at stalled poses, not at the original start. S-bend
and narrow-gate comparisons include interrupted runs; do not count these as
failures or evidence of improvement. Five selectable scenes are not five
completed success-rate comparisons. Results cover static simulation only.

The effective change is observed scanning plus bounded recovery, not a model
benefit. See docs/rule_recovery_web.md, docs/rule_comparison.md and
 docs/five_scene_comparison.md for evidence and implementation details.

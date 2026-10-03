# Offline hierarchical recovery decisions

This experiment changes question organization only. It does not connect to the live observer or execute commands. Planner, strategy generation, timestamps, collision checks and the existing 8-second result TTL remain unchanged.

The first question chooses keep_current, request_new_candidates, uncertain, pivot (turn/probe), or retreat (retreat/turn/probe). The second question contains only the existing members of the chosen family plus the same three non-executing options. It independently checks evidence and can abstain even after a family was selected. Families are derived from actual stage names; unsupported stage layouts and empty families fail closed. No option probabilities are summed or interpreted as safety probabilities.

Replay uses the seven frozen observed-connectivity-v2 snapshots from topology_pairs_20261003. Each snapshot is evaluated twice by both flat and hierarchical questions, alternating request order across snapshots and repetitions. All requests retain the identical state, including the full strategy list, so this tests question organization rather than reduced state information. Other diagnostic questions remain unchanged. Raw requests, responses, source identity, ordering and latency are saved per pair. The reported hierarchy latency sums both HTTP calls; runtime scheduling and freshness overhead are additional.

Run from repository root:

```bash
PYTHONPATH=. python3 scripts/replay_hierarchical.py \
  --snapshots tinynav_temp/topology_pairs_20261003 \
  --output tinynav_temp/hierarchical_pairs_20261003 --repeats 2
```

Validation: 11 targeted tests pass, covering family partition, command/source preservation, abstention at both levels, unchanged auxiliary questions, unsupported and empty-family rejection, and existing observer/recovery/policy behavior.

This small replay can show changed decisions, abstention and latency, but cannot establish arrival rate, escape effectiveness, collision safety or calibrated confidence. Snapshot states were collected under the previous policy; repeated deterministic outcomes are not independent environment trials. An online integration requires evidence of benefit first.

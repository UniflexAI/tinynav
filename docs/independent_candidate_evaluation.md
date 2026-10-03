# Offline independent candidate evaluation

This experiment changes offline assessment only. It leaves planner, observed cells, candidate generation, live observer, web backend, execution and safety checks unchanged. No grade authorizes a navigation command.

## Assessment design

Each request contains one unchanged existing recovery strategy and the shared goal, motion, directional rays, planner report, previous attempts and spatial context. Omitted alternatives are explicitly evaluated in separate requests; their absence must not imply the target is the only available action. Named stage evidence and lossless compact JSON use the previous diagnostic helpers.

Two independent questions are scored for each candidate:

1. Observation evidence grade: blocked=0, partial_observation=1, observed_probe=2, uncertain=unresolved. It checks per-stage blocked/unknown fractions and observed blocked cell count. Historical observed footprint labels are not a safety guarantee or an escape-success estimate. Recent freshness and escape utility are not collapsed into this numerical evidence grade.
2. Probe usefulness: investigate, unsuitable, or uncertain, using the actual motion context, recent observations, goal bearing and previous attempts. Investigate is a hypothesis worth a bounded test, not demonstrated escape. Unknown space, stale evidence or documented repeated failures must not be invented away.

Normal, reversed and cyclically shifted option orders are used, with the candidate submission order also changed. The question instructions and meanings remain identical. The original and reflected snapshots are both tested. Candidate IDs and commands are retained; no new trajectories or measurements are created.

## Consensus rule

The offline selector requires all three complete evaluations of every candidate, categorical agreement for both questions across all three orders, and agreement of the model observation grades with a separately calculated numerical reference. Invalid, missing, unknown or inconsistent grades yield uncertain. At least one observed_probe is required. Multiple highest observation grades remain tied regardless of semantic usefulness scores; usefulness cannot break that observation tie. A single highest observed_probe must also be consistently labeled investigate to produce an advice-only strategy ID. That output explicitly leaves escape usefulness unproven and never bypasses freshness, collision or execution checks.

The numerical observation reference is directly computable from the existing fields; it is not a new sensor or planner. Keeping it outside the model makes grade errors auditable. Letter probabilities are not combined into safety or success probabilities, and no outcome-tuned numeric confidence threshold is used.

## Replay and timing

`replay_independent_candidates.py` freezes the seven existing observed-connectivity-v2 snapshots, mirrors each, preflights every per-candidate/per-order prompt against the deployed slot capacity, compares deployed tokenizer IDs with llama.cpp, saves full prompt texts, and verifies every response usage count against the audited rendering. Full requests, answers and expected numerical grades are saved for each case. It records individual request latency, wall time for one candidate sweep, and wall time for all three sweeps needed to obtain consensus. The consensus wall time is the relevant minimum cost of a future decision; a short per-candidate latency does not mean the whole decision fits the existing 8-second TTL. Runtime queueing and new observation collection costs are additional.

These are repeated assessments of frozen measured states, not independent environment trials, new collision checks or arrival-rate evidence. All grades and consensus outcomes are offline. A test of numerical reading cannot establish whether a recovery actually helps a robot escape.

## Run

From repository root:

```bash
PYTHONPATH=. /home/xiaolefang/workspace/startlux-runtime/venv/bin/python \
  scripts/replay_independent_candidates.py \
  --snapshots tinynav_temp/topology_pairs_20261003 \
  --output tinynav_temp/independent_candidates_20261003 \
  --runtime-source /home/xiaolefang/workspace/StartLux-Decision \
  --model-dir /home/xiaolefang/workspace/startlux-runtime/model-4b-q8
```

The runtime remains unchanged. Original prompts and full replay evidence reside under the output directory; summary.json reports correctness, stability, decisions and latency. Tests cover focused-state preservation, permutation validity, worst-stage grading, invalid/missing evidence, complete agreement, refusal to promote unknown space, ties and absent usefulness support.

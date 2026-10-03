# Local observation screening and model-only usefulness

Offline only. The planner, generated candidates, measured observations, live observer/backend and execution safety checks remain unchanged. No replay issues commands.

`local_candidate_screening.screen` computes the existing numerical observation grade from all named per-stage blocked/unknown fractions and measured blocked-cell count. Only historical observed_probe candidates are sent to the model; blocked, partially observed and malformed evidence are locally excluded and logged. Unknown is never promoted to clear. Candidates and stage commands are not modified. Historical completeness is not current footprint safety. More than two eligible candidates yields refusal rather than taking the first two.

The model answers only the existing probe_usefulness question for each eligible candidate. Its evidence, criterion wording and normal/reversed/rotated option permutations match the previous independent evaluation. Shared goal, motion, rays, planning report, attempts and spatial context remain present. Previously excluded alternatives are not represented as model assessments.

The selector supplies the deterministic local grades to the existing offline consistency rule. Locally excluded candidates carry an explicit unresolved usefulness placeholder solely for that rule; they are not fabricated model responses. Three complete usefulness assessments per eligible candidate must agree. A unique highest historical observation grade plus consistent investigate is necessary for an advice-only strategy ID. Ties and inconsistent answers remain uncertain. Existing execution collision checks, freshness TTL and abort logic are never bypassed. An advice ID does not prove that the action escapes.

Replay uses the same seven measured frozen snapshots and their reflections. All prompt texts are saved and compared against deployed llama.cpp token IDs, bounded by actual slot capacity. Each response token usage must match rendered prompts. No model parameter, TTL or candidate command is changed.

The first replay measures assessment cost. The final e2e replay additionally starts its timer before reading the frozen source, reflection, named-field conversion, local screening, candidate isolation, compact serialization and capacity tokenization; the timer ends after consensus selection. Diagnostic preflight/file export and model/tokenizer startup are excluded, as are live sensor/planner-report collection and external queueing. Thus the measurement is a warm-runtime offline decision cost, not a guarantee for online deadlines. Local preprocessing is logged separately.

Run from repository root using the existing runtime venv:

```bash
PYTHONPATH=. /home/xiaolefang/workspace/startlux-runtime/venv/bin/python \
  scripts/replay_local_screening.py \
  --snapshots tinynav_temp/topology_pairs_20261003 \
  --output tinynav_temp/local_screening_e2e_20261004 \
  --runtime-source /home/xiaolefang/workspace/StartLux-Decision \
  --model-dir /home/xiaolefang/workspace/startlux-runtime/model-4b-q8
```

This keeps an offline diagnostic sweep even for observation-tied candidates to measure model usefulness stability. A production selector could immediately abstain on such ties without calling the model, but that optimization is not integrated here. Frozen repetitions are not independent environment trials. Since all current eligible candidates are pivot sequences, this replay says nothing about model evaluation of fully observed retreat strategies.

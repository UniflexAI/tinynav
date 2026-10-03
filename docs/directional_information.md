# Offline directional evidence diagnostics

This work leaves the planner, candidate generator, observer, web backend, execution and safety guards unchanged. The new helpers are imported only by offline scripts. No replay sends a navigation command.

## Actual input audit

`audit_decision_input.py` uses the deployed package's `from_systemone`, `messages`, `render_ids` and the deployed tokenizer, saves exact chat prompts and token IDs, and compares those IDs with llama.cpp's `/tokenize`. It queries `/props` for actual slot capacity. Runtime source hashes are recorded. Every question is independently rendered with the same evidence, one option letter is read from next-token logits, and temperatures normalize probabilities only over listed letters. The input token usage is the sum of question prompts; it is not one long context. The thinking-off prefix is checked. This model has no generated explanation chain in this path, so probabilities cannot reveal its reasoning.

The seven existing v2 snapshots have 28 independent prompts, 1,873--3,794 tokens each, within the current 4,096-token slot. Tokenizer IDs match and the rendered state includes all recorded fields. Historical runtime logs include explicit overlength rejections; none of these seven audited prompts exceeds capacity. No truncated=1 entries were found in the retained log. This does not claim all prior experiments or arbitrary future states fit.

## Presentation controls

`directional_information.explicit` converts each CSV evidence row to named fields, expands endpoint fields, marks turn direction from existing final yaw, and documents the coordinate conventions and unsigned recent total rotation. It does not invent measurements, convert unknown to clear, change ages, or modify commands. Malformed columns, stage names, fractions and connection values fail closed.

Expanded default-spaced JSON exceeds capacity for some dead-end prompts (up to 4,688 tokens). `compact_state_text` preserves the deployed renderer's array `_index` annotations and removes JSON whitespace only. Compact named-field prompts fit under the same runtime settings. This is separate from changing evidence content.

Three presentations are compared on identical states: legacy object JSON, whitespace-only compact CSV JSON string, and compact named-field JSON string. Questions and strategy commands match. The expanded presentation adds definitions and derived direction labels as well as replacing CSV, so effects cannot be attributed to any single named field.

## Reflection controls

`mirror` reflects world y and robot-relative y, changes signed yaw/bearing and camera angular/control yaw signs, swaps rays, planner representative directions, strategy IDs and attempt directions/points. Absolute accumulated recent rotation and unsigned distances/fractions remain unchanged. Existing strategy commands become their reflected counterfactuals for offline tests only. Rejected candidates are not added. Canonical left/right strategy ordering is maintained and reflection twice restores these snapshots. Generic new state schemas require updating/reviewing the mirror, rather than assuming arbitrary new directional fields are handled.

Reflection of a symmetric scene does not demand a unique opposite action. If only one pivot direction is offered, choosing its reflected counterpart is also weak evidence of direction discrimination. Aggregate reflection consistency counts include unchanged abstention; directional cases are reported separately, with these limitations.

`replay_directional_information.py` preflights every prompt against actual capacity, compares tokenizer IDs, writes all exact prompts, runs each original/reflected presentation twice with rotating request order, and checks response usage equals the audited render lengths. Repeated frozen snapshots are not independent environment trials. It records option distributions and mirror-mapped L1 differences, not calibrated safety probabilities.

## Reading and order controls

`probe_direction_readout.py` separately asks which planner side has the smaller collision rejection fraction (left/right/equal/missing), with an expected answer computed from recorded counts. This checks numerical reading only and cannot establish a safe recovery. It simultaneously scores the same existing alternative question with normal and reversed option order. Option IDs, descriptions, evidence and commands remain identical between the order controls. They run as two independent model questions, not a conversation. Raw requests, responses, prompts and usage checks are retained.

## Run

From repository root, use the existing StartLux runtime venv and its source/model directories:

```bash
PYTHONPATH=. /home/xiaolefang/workspace/startlux-runtime/venv/bin/python \
  scripts/audit_decision_input.py --snapshots tinynav_temp/topology_pairs_20261003 \
  --output tinynav_temp/input_audit_20261003 \
  --runtime-source /home/xiaolefang/workspace/StartLux-Decision \
  --model-dir /home/xiaolefang/workspace/startlux-runtime/model-4b-q8
```

The presentation replay and readout probe accept the same directory arguments; use `scripts/replay_directional_information.py --repeats 2` and `scripts/probe_direction_readout.py` with separate output directories.

Evidence of changed choices alone is insufficient for online integration. Unobserved escape geometry, option sensitivity, single-sided candidate sets and real runtime freshness remain material limits. The current endpoint and TTL are retained.

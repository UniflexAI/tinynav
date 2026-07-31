# Keyframe Quality: Find Confusing Map Keyframes

Offline optimization tool. Finds map keyframes that are **non-discriminative** — they look similar
enough to many unrelated places that they keep getting retrieved as false-positive candidates —
and are worth pruning from a map to improve retrieval precision.

## Idea

Take a map built from **bag1** (`build_map_node.py`), and an independent **bag2** as a probe.
For every frame in bag2, query the map for its top-K retrieval candidates. Each candidate's 3D
position is already known (it's a map keyframe with a known pose) — no ground truth for the
query frame is needed.

If the top-K candidates for a query are all near each other, the retrieval is self-consistent —
they plausibly represent the same real place. If the candidates are geometrically scattered, the
retrieval didn't actually find "the right place" and matched by coincidence (typically perceptual
aliasing: visually repetitive environments — e.g. identical-looking office corridors — that look
alike but aren't). That query is marked **bad**, and every map keyframe that was one of its
candidates gets a strike.

After processing the whole probe bag, each map keyframe has:
- `total_participation`: how many times it showed up as a top-K candidate at all
- `bad_participation`: how many of those times the retrieval was flagged bad
- `badness_ratio = bad_participation / total_participation`

Keyframes with a high badness ratio (and enough participation to be statistically meaningful, not
just 1-2 noisy events) are the ones dragging down map precision — they're candidates for removal.
This mirrors the "stop word" idea in BoW vocabularies: a visual feature/keyframe so generic it
shows up everywhere isn't helping discriminate between places.

## Dispersion metric

Rejected two common choices first:
- **stdev of candidate positions** / **max pairwise distance** — both are wrecked by a single
  outlier candidate. 9-agree-1-off would score as "totally dispersed" even though 90% of the
  candidates agree and the retrieval is probably fine.

Used instead: **largest-cluster fraction** — a simple radius-based greedy clustering. For each
candidate, count how many other candidates are within `--cluster_radius_m` of it; the candidate
with the most neighbors defines the cluster; `cluster_fraction = cluster_size / topk`.
`dispersion = 1 - cluster_fraction`.

A query is "bad" if `dispersion > --dispersion_threshold`. Default `0.4`, i.e. require a
**majority** of the top-K candidates to agree (with the default `topk=5`, at least 3/5 must
cluster together) — the simplest defensible bar: if not even half the candidates agree, the
retrieval is essentially guessing.

## Defaults and why

| Flag | Default | Why |
|---|---|---|
| `--topk` | 5 | This is an offline optimization tool, not a live gate — deliberately more tolerant than production's `relocalization_loop_top_k=3` (`map_node.py`) so it catches borderline cases with a bit more statistical signal, at the cost of being slightly more permissive. |
| `--cluster_radius_m` | 1.0 | "Roughly the same place", not a precise-alignment bar. Matches the coarser end of the 0.5m/1.0m precision tiers already used in `tool/benchmark/map_retrieval_self_consistency.py`. |
| `--dispersion_threshold` | 0.4 | Majority rule: need `>50%` of candidates in one cluster (`>= 3/5` at the default topk). |
| `--min_participation` | 3 | Keyframes barely ever retrieved don't have enough samples to trust a badness ratio computed from them. |

All four are CLI flags — tune them once you've looked at the actual distribution on your data,
don't take the defaults as gospel.

## Usage

```bash
uv run python tool/keyframe_quality/find_confusing_keyframes.py \
  --map_path tinynav_db/maps/map_gt \
  --eval_bag_path tinynav_db/rosbags/bag_1970_01_01_08_09_49 \
  --retrieval_backend dinov2_vlad \
  --out_json tinynav_temp/confusing_keyframes_gt_day.json
```

`--map_path` must already exist (built via `build_map_node.py` from bag1) and, depending on
`--retrieval_backend`, have `vlad_descriptors.db`+`vlad_centres` (see `tool/build_vlad_index.py`
to backfill) or a `bow_index.npz` (auto-built on first use if missing, via
`tool/build_bow_index_for_map.py`'s `build_bow_index`).

## Output

`--out_json` (default `tinynav_temp/confusing_keyframes.json`):
```json
{
  "query_count": ..., "bad_query_count": ..., "bad_query_ratio": ...,
  "flagged_keyframe_count": ...,
  "flagged_keyframes": [
    {"timestamp_ns": ..., "total_participation": ..., "bad_participation": ..., "badness_ratio": ...},
    ...
  ]
}
```
Sorted by `badness_ratio` descending (ties broken by higher `total_participation` — more
confident about keyframes seen more often). Optional `--out_per_query_jsonl` dumps every single
query's candidates + verdict, for spot-checking specific cases.

## Scope note

This version only **detects and reports** confusing keyframes — it does not yet remove them from
the map. Actually pruning would mean rewriting `poses.npy` + the `features`/`depths`/
`vlad_descriptors`/`bow_index` shelves and the `infra1_images_db`/`rgb_images_db` video stores
consistently, which is a separate follow-up once the detection side has been validated against
real data.

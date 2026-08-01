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

### 1. Detect

```bash
uv run python tool/keyframe_quality/find_confusing_keyframes.py \
  --map_path tinynav_db/maps/map_gt \
  --eval_bag_path tinynav_db/rosbags/bag_1970_01_01_08_09_49 \
  --out_json tinynav_temp/confusing_keyframes_gt_day.json \
  --out_per_query_jsonl tinynav_temp/confusing_keyframes_gt_day_per_query.jsonl
```

`--map_path` must already exist (built via `build_map_node.py` from bag1) with the DINOv2 patch
VLAD retrieval index it produces by default (`vlad_descriptors.db` + `vlad_centres`).
`--out_per_query_jsonl` is optional for this step alone, but required by step 2 below.

### 2. Review (visually, in a browser)

```bash
uv run python tool/keyframe_quality/generate_review_page.py \
  --map_path tinynav_db/maps/map_gt \
  --eval_bag_path tinynav_db/rosbags/bag_1970_01_01_08_09_49 \
  --flagged_json tinynav_temp/confusing_keyframes_gt_day.json \
  --per_query_jsonl tinynav_temp/confusing_keyframes_gt_day_per_query.jsonl \
  --out_html tinynav_temp/confusing_keyframes_gt_day_review.html
```

Open the resulting HTML file locally (`file://...`) in a browser. It shows every keyframe that
took part in at least one bad (dispersed) retrieval, worst `badness_ratio` first, with:
- the keyframe's own thumbnail
- up to `--max_examples_per_keyframe` (default 4) of its worst example retrievals: the query
  frame's thumbnail, and every candidate in that top-K set, with a green outline for candidates
  in the agreeing cluster and red for the outliers (the flagged keyframe's own thumbnail is
  marked with a gold highlight so you can see whether *it* was the outlier)

Every flagged keyframe starts **checked**, meaning "will be removed". Uncheck any you judge to be
a false positive after looking at the evidence. Type an output folder name and click "生成裁剪命令"
to get a ready-to-run `prune_map.py` command line for exactly the keyframes still checked.

The page is fully self-contained (thumbnails are embedded as base64 JPEGs) and never touches the
map itself or any external server — it only generates a command for you to run.

### 3. Prune (writes a new map, source map untouched)

```bash
uv run python tool/keyframe_quality/prune_map.py \
  --map_path tinynav_db/maps/map_gt \
  --output_path tinynav_db/maps/map_gt_pruned \
  --exclude_timestamps 449729891255,440779900251,...
```

Copies the whole map directory to `--output_path`, then removes the given timestamps from
`poses.npy` and the per-keyframe shelve stores (`features`/`depths`/`vlad_descriptors`/
`embeddings`/`semantic_embeddings`/`patch_tokens`). The source map is never modified.
`map_node.py`/`build_map_node.py` derive the keyframe set entirely from `poses.npy`'s keys, so
this is sufficient — the pruned keyframes stop being used for relocalization.

Video stores (`infra1_images_db`/`rgb_images_db`) are left untouched: nothing reads a keyframe's
image once its timestamp is gone from `poses.npy`, so the orphaned frames are harmless dead
weight. Writes `prune_report.json` into the output folder with before/after keyframe counts and
per-store removal counts.

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
confident about keyframes seen more often). Note this list includes *every* keyframe meeting
`--min_participation`, not just ones that were actually flagged bad — many will have
`badness_ratio == 0`. `generate_review_page.py` filters to `bad_participation > 0` before
building the review page. `--out_per_query_jsonl` dumps every single query's candidates +
verdict, for spot-checking specific cases.

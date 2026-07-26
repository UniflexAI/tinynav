# TinyNav Mapping Benchmark

This benchmark module evaluates the accuracy and consistency of TinyNav's mapping and localization capabilities.

## Benchmark Pipeline 1: Cross-Map Localization Accuracy

This benchmark evaluates how well TinyNav can map and localize using two datasets by a self-consistency approach. For best results, ensure test datasets contain enough trajectory overlap.

### Process Overview

1. **Map A Creation**: Run mapping on ROS bag A to create map A
2. **Localization in Map A**: Use map A to localize poses from ROS bag B (100 selected images)
3. **Map B Creation**: Run mapping on ROS bag B to create map B (ground truth)
4. **Pose Extraction**: Extract ground truth poses from map B for the same 100 timestamps
5. **Transformation Estimation**: Use RANSAC to estimate rigid transformation between map A and map B coordinate systems
6. **Accuracy Evaluation**: Compute localization accuracy with multiple precision thresholds

### Image Selection Strategy

From each ROS bag, 100 images are selected for evaluation:
- Remove front and back 5% of trajectory when robot is likely stationary
- Select images evenly spaced in time from the remaining 90% of trajectory
- This ensures evaluation on dynamic, representative motion data

### Evaluation Metrics

Localization accuracy is measured with three precision categories:
- **High Precision**: Within 5cm translation / 2° rotation
- **Medium Precision**: Within 10cm translation / 5° rotation
- **Low Precision**: Within 30cm translation / 10° rotation

Results are reported as percentage of the 100 test images meeting each precision threshold.

### Coordinate System Alignment

Since map A and map B are built independently, they exist in different coordinate systems. The benchmark:
1. Uses pose pairs (localization result from map A, ground truth from map B) for the same 100 timestamps
2. Applies RANSAC to estimate the optimal rigid transformation between coordinate systems
3. Transforms all map A localization results to map B coordinate system for comparison
4. Treats map B poses as ground truth for evaluation

### Usage

```bash
# Run benchmark between two ROS bags
uv run python tool/benchmark/benchmark_mapping.py --bag_a path/to/bag_a.db3 --bag_b path/to/bag_b.db3 --output_dir results/

# Example with provided test data
uv run python tool/benchmark/benchmark_mapping.py \
    --bag_a my_bag_a \
    --bag_b my_bag_b \
    --output_dir benchmark_results/
```

#### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bag_a` | Required | Path to ROS bag A (for creating reference map) |
| `--bag_b` | Required | Path to ROS bag B (for localization and ground truth) |
| `--output_dir` | `output/benchmark_results` | Output directory for results |
| `--num_images` | `100` | Number of evaluation samples |
| `--rate` | `1.0` | Playback rate for ROS bags (e.g., 2.0 for 2x speed) |
| `--timeout` | `60` | Timeout for each mapping process (seconds) |
| `--verbose_timer` | - | Enable verbose timer output for all nodes |
| `--no_verbose_timer` | - | Disable verbose timer output (cleaner logs) |

#### Timer Output Control

By default, the benchmark runs in quiet mode with minimal timer output for cleaner logs. You can control the verbosity:

```bash
# Default behavior - quiet mode (recommended for production)
uv run python tool/benchmark/benchmark_mapping.py --bag_a bag_a --bag_b bag_b

# Enable verbose timing for debugging/development
uv run python tool/benchmark/benchmark_mapping.py --bag_a bag_a --bag_b bag_b --verbose_timer

# Explicit quiet mode (same as default)
uv run python tool/benchmark/benchmark_mapping.py --bag_a bag_a --bag_b bag_b --no_verbose_timer
```

**Note**: Verbose timer mode shows detailed timing information from perception, mapping, and localization nodes, which can be helpful for performance analysis but produces significantly more log output.

### Output

The benchmark generates:
- **Summary Report**: Overall accuracy statistics and performance metrics
- **Detailed Results**: Per-image localization errors and success/failure analysis
- **Transformation Matrix**: Estimated rigid transformation between coordinate systems
- **Visualization Data**: Trajectory plots and error distributions (if applicable)


## Benchmark Pipeline 2: Cross-Map Self-Consistency Evaluation (VLAD)

`map_retrieval_self_consistency.py` evaluates DINOv2 patch VLAD retrieval (the descriptor TinyNav actually ships and uses at runtime, see `tinynav/core/vlad.py`) across two independently-built maps **without needing a known coordinate transform between them**. Map B keyframes query Map A; the script fits its own SE(2) transform from the retrieval's own top-1 matches via RANSAC, then measures how consistent the rest of the top-k candidates are with that fitted transform.

It always reads `vlad_descriptors.db` from both maps — there is no descriptor-backend switch, matching the fact that TinyNav's map-building and relocalization code (`build_map_node.py` / `map_node.py`) always uses VLAD with no other option (see the PR's "Notes" section).

### Why self-consistency instead of a known transform

Map A and Map B are each built independently (separate mapping runs), so their coordinate frames don't line up. There is no ready-made ground-truth transform between them, and getting one (e.g. manual alignment, a fixed marker) is extra setup. Fitting the transform from the retrieval's own top-1 matches avoids that setup — the trade-off is that the fitted transform is only as good as the retrieval being evaluated, so treat the resulting metrics as a **self-consistency signal**, not external ground-truth accuracy.

### Test dataset

The GT/day/night ROS bags used to produce the results below are published at
[`UniflexAI/rosbag_tinynav_vlad_eval`](https://huggingface.co/datasets/UniflexAI/rosbag_tinynav_vlad_eval)
on Hugging Face (same download pattern as `UniflexAI/rosbag2_go2_looper` in
`scripts/run_rosbag_build_map.sh`):

```bash
hf download --repo-type dataset UniflexAI/rosbag_tinynav_vlad_eval --local-dir /tinynav/tinynav_db/rosbags
```

| Bag | Role |
|---|---|
| `bag_gt` | GT map (reference / retrieval database) |
| `bag_day` | Day query map |
| `bag_night` | Night query map |

### Usage

```bash
uv run python tool/benchmark/map_retrieval_self_consistency.py \
  --map-a /tinynav/output/map_gt \
  --map-b /tinynav/output/map_day \
  --output-dir /tinynav/output/self_consistency_vlad
```

| Option | Default | Description |
|---|---|---|
| `--topk` | `1,3,5,10` | Comma-separated top-K values to report recall/precision/IoU for. |
| `--distance-thresholds` | `0.5,1.0` | Comma-separated hit-radius thresholds in meters. |
| `--ransac-threshold-m` | `0.5` | Inlier threshold (meters) for the SE(2) RANSAC fit. |
| `--ransac-iterations` | `3000` | RANSAC iterations for the fit. |
| `--every-n` | `1` | Evaluate every Nth Map B keyframe (use >1 to subsample for a quick check). |
| `--max-queries` | `0` | Cap on number of Map B queries (`0` = no cap). |
| `--seed` | `7` | RNG seed for RANSAC sampling. |

### Output

- `summary.json`: overall metrics, the fitted SE(2) transform, and top-1 residual stats (mean/median/p90).
- `metrics.csv`: recall / precision / IoU for each top-K × distance-threshold combination.
- `per_query_results.jsonl`: per-query retrieved candidates and the fitted-transform residual.

Recommended primary score: `top1_inlier_ratio["0.5m"]` from `summary.json`.

### Example results

Run against the GT/day/night bags above, comparing VLAD (this branch) against the DINOv2 global-embedding baseline from main. See the `feat(vlad): ...` PR description for the full write-up, including why the three VLAD training variants are grouped together (their differences aren't a reliable signal, see "Why batched, not strict single-point" there).

#### Day: `map_day -> map_gt`

| Method | Fit Inlier@1m | Top1 Mean | Top1 Median | Top1 P90 | R@1 0.5m | R@10 0.5m | IoU@10 0.5m |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 global | 92.03% | 0.433m | 0.229m | 0.839m | 78.80% | 97.44% | 0.3325 |
| DINOv2 patch VLAD (original, in-memory) | 98.80% | 0.270m | 0.197m | 0.505m | 89.62% | 99.10% | 0.4161 |
| + disk-persisted online k-means (strict single-point) | 99.10% | 0.232m | 0.180m | 0.413m | 93.08% | 99.25% | 0.4364 |
| + disk-persisted online k-means (batched, **shipped**) | 98.05% | 0.304m | 0.194m | 0.509m | 88.87% | 99.25% | 0.4163 |

#### Night: `map_night -> map_gt`

| Method | Fit Inlier@1m | Top1 Mean | Top1 Median | Top1 P90 | R@1 0.5m | R@10 0.5m | IoU@10 0.5m |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 global | 29.31% | 8.698m | 8.099m | 22.559m | 23.34% | 35.96% | 0.0996 |
| DINOv2 patch VLAD (original, in-memory) | 35.14% | 6.885m | 7.421m | 16.955m | 31.48% | 39.08% | 0.1422 |
| + disk-persisted online k-means (strict single-point) | 35.28% | 6.568m | 7.271m | 16.579m | 31.75% | 37.04% | 0.1515 |
| + disk-persisted online k-means (batched, **shipped**) | 36.64% | 6.626m | 7.382m | 16.701m | 31.21% | 39.89% | 0.1407 |

All three DINOv2 patch VLAD variants clearly outperform the DINOv2 global baseline on both day and night.

## Future Benchmark Pipelines

Additional benchmark pipelines will be added to evaluate:
- Long-term map consistency and drift
- Multi-session mapping accuracy
- Computational performance and resource usage
- Robustness to different lighting and environmental conditions

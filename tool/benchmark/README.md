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


## Future Benchmark Pipelines

Additional benchmark pipelines will be added to evaluate:
- Long-term map consistency and drift
- Multi-session mapping accuracy
- Computational performance and resource usage
- Robustness to different lighting and environmental conditions


import numpy as np
import time
import sys
import os
from numba import njit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from planning_node import run_raycasting_loopy
from tinynav.tinynav_cpp_bind import run_raycasting_cpp

@njit
def run_raycasting(depth_image, T_cam_to_world, grid_shape, fx, fy, cx, cy, origin, step, resolution):
    occupancy_grid = np.zeros(grid_shape)
    depth_height, depth_width = depth_image.shape
    for v in range(0, depth_height, step):
        for u in range(0, depth_width, step):
            d = depth_image[v, u]
            if d <= 0:
                continue
            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            z = d
            point_cam = np.array([x, y, z, 1.0])
            point_world = T_cam_to_world @ point_cam
            camera_origin = T_cam_to_world[:3, 3]
            start_voxel = np.floor((camera_origin - origin) / resolution).astype(np.int32)
            end_voxel = np.floor((point_world[:3] - origin) / resolution).astype(np.int32)
            diff = end_voxel - start_voxel
            steps = np.max(np.abs(diff))
            if steps == 0:
                continue
            for i in range(steps + 1):
                t = i / steps
                interp = np.round(start_voxel + t * diff).astype(np.int32)
                if np.any(interp < 0) or np.any(interp >= np.array(grid_shape)):
                    continue
                x, y, z = interp[0], interp[1], interp[2]
                occupancy_grid[x, y, z] -= 0.05
            if np.all(end_voxel >= 0) and np.all(end_voxel < np.array(grid_shape)):
                x, y, z = end_voxel[0], end_voxel[1], end_voxel[2]
                occupancy_grid[x, y, z] += 0.2
    #clip the occupancy grid to [-5, 10]
    occupancy_grid = np.clip(occupancy_grid, -0.1, 0.1)
    return occupancy_grid

def print_diffs(arr1, arr2, name1, name2):
    """Helper function to print differences between two arrays."""
    print(f"\nERROR: Outputs of {name1} and {name2} implementations do not match.")
    diff = np.abs(arr1 - arr2)
    
    num_diffs_to_show = 5
    num_diffs = np.count_nonzero(diff > 1e-6) # Count non-trivial differences
    if num_diffs < num_diffs_to_show:
        num_diffs_to_show = num_diffs

    if num_diffs_to_show == 0:
        print("No significant non-zero differences found, but np.allclose failed. This might be a tolerance issue or very small floating point discrepancies.")
        return

    flat_diff_indices = np.argsort(diff.flatten())[-num_diffs_to_show:][::-1]
    
    print(f"\n--- Top {num_diffs_to_show} Largest Differences ---")
    for flat_idx in flat_diff_indices:
        idx = np.unravel_index(flat_idx, diff.shape)
        print(f"Index: {idx}")
        print(f"  {name1} value: {arr1[idx]:.8f}")
        print(f"  {name2} value:    {arr2[idx]:.8f}")
        print(f"  Difference:   {diff[idx]:.8f}")
        print("-" * 20)

def test_run_raycasting_comparison():
    # Test parameters
    grid_shape = (100, 20, 100)
    resolution = 0.1
    origin = np.array(grid_shape) * resolution / -2.
    step = 10
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    T_cam_to_world = np.eye(4)
    
    # Create a sample depth image
    depth_height, depth_width = 480, 640
    depth_image = 4.0 * np.ones((depth_height, depth_width), dtype=np.float32)

    # --- Python (Numba, Vectorized) version ---
    print("--- Benchmarking Python (Numba, Vectorized) ---")
    print("Warming up Numba JIT...")
    run_raycasting(depth_image, T_cam_to_world, grid_shape, fx, fy, cx, cy, origin, step, resolution)
    print("Warmup complete.")

    num_runs = 10
    py_timings = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        py_occupancy_grid = run_raycasting(depth_image, T_cam_to_world, grid_shape, fx, fy, cx, cy, origin, step, resolution)
        end_time = time.perf_counter()
        py_timings.append(end_time - start_time)

    avg_py_time_ms = (sum(py_timings) / num_runs) * 1000
    print(f"Avg execution time: {avg_py_time_ms:.2f} ms")
    print(f"Result sum: {np.sum(py_occupancy_grid)}")

    # --- Python (Numba, Loopy) version ---
    print("\n--- Benchmarking Python (Numba, Loopy) ---")
    print("Warming up Numba JIT...")
    run_raycasting_loopy(depth_image, T_cam_to_world, grid_shape, fx, fy, cx, cy, origin, step, resolution)
    print("Warmup complete.")

    loopy_timings = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        loopy_occupancy_grid = run_raycasting_loopy(depth_image, T_cam_to_world, grid_shape, fx, fy, cx, cy, origin, step, resolution)
        end_time = time.perf_counter()
        loopy_timings.append(end_time - start_time)
    avg_loopy_time_ms = (sum(loopy_timings) / num_runs) * 1000
    print(f"Avg execution time: {avg_loopy_time_ms:.2f} ms")
    print(f"Result sum: {np.sum(loopy_occupancy_grid)}")

    # --- C++ (pybind11) version ---
    print("\n--- Benchmarking C++ (pybind11) ---")
    cpp_timings = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        cpp_occupancy_grid_flat = run_raycasting_cpp(depth_image, T_cam_to_world, list(grid_shape), fx, fy, cx, cy, origin, step, resolution)
        end_time = time.perf_counter()
        cpp_timings.append(end_time - start_time)
    cpp_occupancy_grid = cpp_occupancy_grid_flat.reshape(grid_shape)
    avg_cpp_time_ms = (sum(cpp_timings) / num_runs) * 1000
    print(f"Avg execution time: {avg_cpp_time_ms:.2f} ms")
    print(f"Result sum: {np.sum(cpp_occupancy_grid)}")

    # --- Verification ---
    print("\n--- Verifying Results ---")
    vectorized_vs_loopy = np.allclose(py_occupancy_grid, loopy_occupancy_grid, atol=1e-5)
    vectorized_vs_cpp = np.allclose(py_occupancy_grid, cpp_occupancy_grid, atol=1e-5)

    if vectorized_vs_loopy and vectorized_vs_cpp:
        print("Success! All implementations produce consistent results.")
    else:
        if not vectorized_vs_loopy:
            print_diffs(py_occupancy_grid, loopy_occupancy_grid, "Vectorized", "Loopy")
        if not vectorized_vs_cpp:
            print_diffs(py_occupancy_grid, cpp_occupancy_grid, "Vectorized", "C++")
        assert False, "Implementations do not match."

if __name__ == "__main__":
    test_run_raycasting_comparison()


# --- climb region: per-cell relaxation of the obstacle span filter -----------

def _riser_grid(height_m, config, resolution=0.05):
    """3x3 cells; the centre one holds occupancy from the ground up to height_m."""
    z_layers = int(round((config.robot_z_top - config.robot_z_bottom) / resolution))
    grid = np.zeros((3, 3, z_layers))
    grid[1, 1, :int(round(height_m / resolution))] = 0.2
    return grid, np.array([0.0, 0.0, config.robot_z_bottom])


def _span_map(config, relaxed_cell, relaxed=0.3):
    m = np.full((3, 3), config.min_wall_span_m)
    m[relaxed_cell] = relaxed
    return m


def test_min_span_map_relaxes_only_the_cells_it_covers():
    from planning_node import ObstacleConfig, build_obstacle_map
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.15, config)
    args = (grid, origin, 0.05)
    assert build_obstacle_map(*args, robot_z=0.0, config=config)[1, 1]
    assert not build_obstacle_map(*args, robot_z=0.0, config=config,
                                  min_span_map=_span_map(config, (1, 1)))[1, 1]
    # relaxing somewhere else must not relax this cell -- the whole point of going
    # per-cell instead of switching the threshold globally
    assert build_obstacle_map(*args, robot_z=0.0, config=config,
                              min_span_map=_span_map(config, (0, 0)))[1, 1]


def test_min_span_map_still_blocks_taller_verticals():
    from planning_node import ObstacleConfig, build_obstacle_map
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.35, config)   # above the relaxed threshold
    assert build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config,
                              min_span_map=_span_map(config, (1, 1)))[1, 1]


def test_min_span_map_relaxes_steps_above_the_ground_band():
    """Mid-climb the next riser starts above robot_z_bottom + ground_band_m; a relaxed
    cell must still read it as a step, not as a floating obstacle."""
    from planning_node import ObstacleConfig, build_obstacle_map
    config = ObstacleConfig()
    resolution = 0.05
    grid, origin = _riser_grid(0.0, config, resolution)
    base = config.ground_band_m + resolution        # above the ground band
    lo = int(round(base / resolution))
    hi = int(round((base + 0.15) / resolution))     # 0.15m riser, below the relaxed threshold
    grid[1, 1, lo:hi] = 0.2
    args = (grid, origin, resolution)
    assert build_obstacle_map(*args, robot_z=0.0, config=config)[1, 1]
    assert not build_obstacle_map(*args, robot_z=0.0, config=config,
                                  min_span_map=_span_map(config, (1, 1)))[1, 1]


def test_no_min_span_map_is_the_strict_default():
    from planning_node import ObstacleConfig, build_obstacle_map
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.15, config)
    assert np.array_equal(
        build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config),
        build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config,
                           min_span_map=None))

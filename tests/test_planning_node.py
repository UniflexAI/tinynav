
import numpy as np
import time
import sys
import os
from numba import njit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from planning_node import run_raycasting_loopy, generate_trajectory_library_3d, goal_heading_error
from tinynav.core.math_utils import matrix_to_quat
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

# Robot at the world origin, body +Z (forward) along world +X, body +Y (down) along
# world -Z -- the camera convention /slam/odometry_visual publishes.
_FACING_X = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])

def _pick(target, heading_weight):
    """Lowest-cost (vx, omega) for a goal in open space.

    Mirrors sync_callback's cost_function with an all-clear ESDF (score == 0) and
    no reverse gating, so only the distance / heading / smoothness terms decide.
    Keep the weights in sync with planning_node if they change there.
    """
    trajectories, params = generate_trajectory_library_3d(
        init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X),
    )
    last_param = np.zeros(2)  # previous pick was a full stop
    costs = [
        100 * np.linalg.norm(trajectories[i][-1, :3] - target)
        + heading_weight * goal_heading_error(trajectories[i][-1], target)
        + 10 * abs(last_param[0] - params[i, 0])
        + 10 * abs(last_param[1] - params[i, 1])
        for i in range(len(trajectories))
    ]
    return params[np.argsort(costs, kind='stable')[0]]

def test_goal_heading_error():
    end = np.empty(7)
    end[:3] = 0.0
    end[3:] = matrix_to_quat(_FACING_X)
    for target, expected in (
        ([5.0, 0.0, 0.0], 0.0),          # dead ahead
        ([-5.0, 0.0, 0.0], np.pi),       # dead behind
        ([0.0, 5.0, 0.0], np.pi / 2),    # abeam
        ([0.0, -5.0, 0.0], np.pi / 2),   # abeam, other side -- error is unsigned
        ([0.0, 0.0, 5.0], 0.0),          # straight up: no XY bearing, no opinion
    ):
        got = goal_heading_error(end, np.array(target))
        assert abs(got - expected) < 1e-6, f"target {target}: {got} != {expected}"
    print("goal_heading_error: ok")

def test_goal_behind_turns_in_place():
    """The regression: forward-only samples all recede from a goal behind the robot,
    so distance alone picks vx=0, and identical vx=0 endpoints leave smoothness to
    pick omega=0 too. The robot must turn around instead of stalling."""
    behind = np.array([-5.0, 0.0, 0.0])
    vx_old, omega_old = _pick(behind, heading_weight=0.0)
    assert (vx_old, omega_old) == (0.0, 0.0), f"expected the stall, got {vx_old}, {omega_old}"

    vx, omega = _pick(behind, heading_weight=100.0)
    assert abs(omega) > 1e-6, f"goal behind still yields omega={omega}"
    print(f"goal behind: vx={vx:.3f} omega={omega:.3f} (was vx=0 omega=0)")

def test_goal_ahead_still_drives_straight():
    vx, omega = _pick(np.array([5.0, 0.0, 0.0]), heading_weight=100.0)
    assert vx > 0.0 and abs(omega) < 1e-6, f"goal ahead yields vx={vx}, omega={omega}"
    print(f"goal ahead: vx={vx:.3f} omega={omega:.3f}")

def test_goal_abeam_turns_while_driving():
    for side in (5.0, -5.0):
        vx, omega = _pick(np.array([0.0, side, 0.0]), heading_weight=100.0)
        assert vx > 0.0 and abs(omega) > 1e-6, f"goal abeam yields vx={vx}, omega={omega}"
        print(f"goal abeam ({side:+.0f}): vx={vx:.3f} omega={omega:.3f}")

if __name__ == "__main__":
    test_goal_heading_error()
    test_goal_behind_turns_in_place()
    test_goal_ahead_still_drives_straight()
    test_goal_abeam_turns_while_driving()
    test_run_raycasting_comparison()

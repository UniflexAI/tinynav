
import numpy as np
import time
import sys
import os
from numba import njit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from planning_node import (
    run_raycasting_loopy,
    generate_trajectory_library_3d,
    generate_two_stage_trajectory_library_3d,
    goal_heading_error,
    reverse_gate_hysteresis,
)
from scipy.spatial.transform import Rotation as R
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

# body +Z (forward) along world +X, body +Y (down) along world -Z: the camera convention
_FACING_X = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])

# mirrors the regular-trajectory term of PlanningNode.cost_function (planning_node.py); keep
# the weights and heading fade distance (2.0) in sync by hand
_ESDF_WEIGHT = 2000.0
_DIST_WEIGHT = 1000.0
_HEADING_WEIGHT = 100.0
_HEADING_FADE_DIST = 2.0
_CMD_VEL_MAX_LINEAR_ACC = 0.6
_DECEL_WEIGHT = 500.0

def _trajectory_cost(traj, param, score, target_end, last_param, heading_weight=_HEADING_WEIGHT):
    dist = np.linalg.norm(np.asarray(traj[-1, :3]) - target_end)
    heading = goal_heading_error(traj[-1], target_end) * min(1.0, dist / _HEADING_FADE_DIST)
    return (
        score * _ESDF_WEIGHT
        + _DIST_WEIGHT * dist
        + heading_weight * heading
        + 10 * abs(last_param[0] - param[0])
        + 10 * abs(last_param[1] - param[1])
    )

def _two_stage_trajectory_cost(traj, param, score, target_end, last_param, heading_weight=_HEADING_WEIGHT, planning_dt=0.1):
    """Mirrors PlanningNode.cost_function's two-stage variant: blends the
    stage-1 endpoint (what's actually executed) with the full-horizon endpoint
    (a softer lookahead), instead of scoring the full-horizon endpoint alone -
    otherwise the planner can "plan" progress into a stage 2 that never
    actually runs, picking vx1=0 every cycle and never moving. Also penalizes
    requesting more deceleration than CmdVelControlNode's rate limiter can
    actually deliver in one planning cycle, which otherwise lets the planner
    coast at speed right up to an obstacle and overshoot while it brakes."""
    stage1_end_idx = len(traj) // 2 - 1

    def dist_and_heading(pose):
        d = np.linalg.norm(np.asarray(pose[:3]) - target_end)
        h = goal_heading_error(pose, target_end) * min(1.0, d / _HEADING_FADE_DIST)
        return d, h

    dist_final, heading_final = dist_and_heading(traj[-1])
    dist_mid, heading_mid = dist_and_heading(traj[stage1_end_idx])
    dist = 0.5 * (dist_final + dist_mid)
    heading = 0.5 * (heading_final + heading_mid)
    feasible_dv = _CMD_VEL_MAX_LINEAR_ACC * max(planning_dt, 0.03)
    infeasible_decel = max(0.0, (last_param[0] - param[0]) - feasible_dv)
    return (
        score * _ESDF_WEIGHT
        + _DIST_WEIGHT * dist
        + heading_weight * heading
        + 10 * abs(last_param[0] - param[0])
        + 10 * abs(last_param[1] - param[1])
        + _DECEL_WEIGHT * infeasible_decel
    )

def _pick(target, heading_weight=_HEADING_WEIGHT):
    """Lowest-cost (vx, omega) from the planner's own cost terms, with a clear ESDF
    (score=0), no reverse gating and a standing start."""
    trajectories, params = generate_trajectory_library_3d(init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X))
    last_param = np.zeros(2)
    costs = [
        _trajectory_cost(trajectories[i], params[i], 0.0, target, last_param, heading_weight=heading_weight)
        for i in range(len(trajectories))
    ]
    return params[np.argsort(costs, kind='stable')[0]]

def test_goal_heading_error():
    end = np.zeros(7)
    end[3:] = matrix_to_quat(_FACING_X)
    for target, expected in (
        ([5.0, 0.0, 0.0], 0.0),
        ([-5.0, 0.0, 0.0], np.pi),
        ([0.0, 5.0, 0.0], np.pi / 2),
        ([0.0, -5.0, 0.0], np.pi / 2),  # unsigned, so both sides score the same
        ([0.0, 0.0, 5.0], 0.0),         # no XY bearing
    ):
        got = goal_heading_error(end, np.array(target))
        assert abs(got - expected) < 1e-6, f"target {target}: {got} != {expected}"

def test_goal_behind_curves_toward_target():
    # forward-only samples can't reverse, but a tight max-rate turn still shaves a little
    # off the final distance to a target directly behind; at _DIST_WEIGHT=1000 that small
    # gain outweighs the smoothness cost of moving, so the pick curves instead of standing
    # still or rotating in place
    behind = np.array([-5.0, 0.0, 0.0])
    vx0, omega0 = _pick(behind, heading_weight=0.0)
    assert vx0 > 0.0 and abs(omega0) > 1e-6, f"target behind (no heading term) yields vx={vx0}, omega={omega0}"

    vx, omega = _pick(behind)
    assert vx > 0.0 and abs(omega) > 1e-6, f"target behind still yields vx={vx}, omega={omega}"

def test_goal_ahead_still_drives_straight():
    vx, omega = _pick(np.array([5.0, 0.0, 0.0]))
    assert vx > 0.0 and abs(omega) < 1e-6, f"target ahead yields vx={vx}, omega={omega}"

def test_goal_abeam_turns_while_driving():
    for side in (5.0, -5.0):
        vx, omega = _pick(np.array([0.0, side, 0.0]))
        assert vx > 0.0 and abs(omega) > 1e-6, f"target abeam yields vx={vx}, omega={omega}"

def test_heading_fades_within_arrival_radius():
    # inside HEADING_FADE_DIST the heading term is scaled down, so it cannot outbid distance:
    # the pick must still be the trajectory that lands closest to the goal
    trajectories, params = generate_trajectory_library_3d(init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X))
    close = np.array([0.4, 0.3, 0.0])
    assert np.linalg.norm(close) < _HEADING_FADE_DIST

    dists = [np.linalg.norm(trajectories[i][-1, :3] - close) for i in range(len(trajectories))]
    nearest = int(np.argmin(dists))
    picked = _pick(close)
    assert tuple(picked) == tuple(params[nearest]), f"close goal picked {picked}, not nearest {params[nearest]}"

def test_heading_fade_is_monotonic_in_distance():
    # same bearing error, different range: the heading penalty grows with distance and saturates
    end = np.zeros(7)
    end[3:] = matrix_to_quat(_FACING_X)
    traj = end[None, :]
    last_param = np.zeros(2)
    param = np.zeros(2)

    def heading_term(range_m):
        target = np.array([0.0, range_m, 0.0])  # 90 deg off the nose at every range
        return _trajectory_cost(traj, param, 0.0, target, last_param) - _DIST_WEIGHT * range_m

    terms = [heading_term(r) for r in (0.5, 1.0, 2.0, 4.0)]
    assert terms[0] < terms[1] < terms[2], f"heading penalty not growing with range: {terms}"
    assert abs(terms[2] - terms[3]) < 1e-9, f"heading penalty not saturated past the fade distance: {terms}"
    assert abs(terms[2] - _HEADING_WEIGHT * np.pi / 2) < 1e-9, f"saturated penalty {terms[2]} != full weight"

def _pick_two_stage(target, heading_weight=_HEADING_WEIGHT):
    """Same selection logic as _pick, but over the two-stage lattice, using
    _two_stage_trajectory_cost (mirrors PlanningNode.cost_function)."""
    trajectories, params = generate_two_stage_trajectory_library_3d(init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X))
    last_param = np.zeros(4)
    costs = [
        _two_stage_trajectory_cost(trajectories[i], params[i], 0.0, target, last_param, heading_weight=heading_weight)
        for i in range(len(trajectories))
    ]
    return params[np.argsort(costs, kind='stable')[0]]

def test_two_stage_trajectory_count_and_shape():
    trajectories, params = generate_two_stage_trajectory_library_3d(init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X))
    single_trajectories, _ = generate_trajectory_library_3d(init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X))
    assert params.shape == (trajectories.shape[0], 4), "params must carry (vx1, omega1, vx2, omega2)"
    assert trajectories.shape[0] == 107, "default 2x5 squared (100) plus 7 coast speeds, should be 107"
    # candidate count should stay in the same ballpark as the single-stage lattice
    # it replaced, so ESDF scoring cost doesn't regress on-robot (a finer vx grid
    # isn't needed for arrival precision - the blended stage-1/full-horizon cost
    # in PlanningNode.cost_function handles that instead, see
    # test_two_stage_intermediate_distance_still_moves)
    assert 0.5 <= trajectories.shape[0] / single_trajectories.shape[0] <= 2.0

def test_two_stage_dynamic_window_bounds_stage1_speed():
    # only stage 1 is ever executed, so every candidate's vx1 must be reachable
    # from the robot's current speed within one planning cycle - otherwise the
    # planner can request a speed change the real acceleration limit can't
    # deliver in time (this is what caused corner overshoot: see the
    # feasible_dv commit history)
    current_vx, max_vx_step = 0.3, 0.072
    trajectories, params = generate_two_stage_trajectory_library_3d(
        init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X),
        max_linear_vel=0.5, max_angular_vel=0.75,
        current_vx=current_vx, max_vx_step=max_vx_step,
    )
    vx1 = params[:, 0]
    assert np.all(vx1 >= current_vx - max_vx_step - 1e-9)
    assert np.all(vx1 <= current_vx + max_vx_step + 1e-9)

    # default (no current_vx/max_vx_step given) stays unconstrained, so callers
    # that don't track a "current speed" - e.g. the tests above - see the same
    # full [0, max_linear_vel] behavior as before this change
    _, unconstrained_params = generate_two_stage_trajectory_library_3d(
        init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X),
        max_linear_vel=0.5, max_angular_vel=0.75,
    )
    assert unconstrained_params[:, 0].max() == 0.5

def test_two_stage_expresses_straight_then_turn():
    # a shape a single constant-curvature arc cannot produce: no drift during
    # stage 1, then a sharp turn during stage 2
    trajectories, params = generate_two_stage_trajectory_library_3d(init_p=np.zeros(3), init_q=matrix_to_quat(_FACING_X))
    straight_then_turn = np.where(
        (params[:, 1] == 0.0) & (params[:, 0] > 0.0) & (np.abs(params[:, 3]) > 0.5)
    )[0]
    assert len(straight_then_turn) > 0, "expected at least one straight->turn candidate in the lattice"

    idx = straight_then_turn[0]
    traj = trajectories[idx]
    # traj[i] is the pose after (i+1) integration steps (see the two-stage
    # generator), so the last stage-1 sample sits at the midpoint minus one.
    stage1_end = len(traj) // 2 - 1

    def yaw(pose):
        return R.from_quat(pose[3:7]).as_euler("xyz")[2]

    # heading barely changes during the straight stage 1, then swings sharply
    # during the turning stage 2 - a single constant-curvature arc can't do this
    heading_change_stage1 = abs(yaw(traj[stage1_end]) - yaw(traj[0]))
    heading_change_stage2 = abs(yaw(traj[-1]) - yaw(traj[stage1_end]))
    assert heading_change_stage1 < 1e-6
    assert heading_change_stage2 > 0.5

def test_two_stage_goal_ahead_drives_straight_both_stages():
    vx1, omega1, vx2, omega2 = _pick_two_stage(np.array([5.0, 0.0, 0.0]))
    assert vx1 > 0.0 and abs(omega1) < 1e-6, f"target ahead yields vx1={vx1}, omega1={omega1}"
    assert vx2 > 0.0 and abs(omega2) < 1e-6, f"target ahead yields vx2={vx2}, omega2={omega2}"

def test_two_stage_goal_abeam_turns_while_driving():
    for side in (5.0, -5.0):
        vx1, omega1, vx2, omega2 = _pick_two_stage(np.array([0.0, side, 0.0]))
        assert vx1 > 0.0, f"target abeam yields vx1={vx1}"
        assert abs(omega1) > 1e-6 or abs(omega2) > 1e-6, "expected some turning toward an abeam target"

def test_two_stage_intermediate_distance_still_moves():
    # regression: a coarse {0, max} vx grid can only land the horizon's end
    # point at ~0, half, or full range, so a target between those gets
    # "rounded" to whichever is numerically closest every cycle. If that's the
    # near-zero one, the plan always starts with vx1=0 - and since the
    # executed command only ever covers stage 1 (see PlanningNode.
    # cost_function's smoothness term, which compares against the previous
    # stage-1 pick), the robot never actually moves even though a forward
    # target is trivially reachable. A finer vx grid must keep vx1 > 0 across
    # the whole reachable range, not just at its extremes.
    # very close targets (<~0.5m) are excluded: there ties legitimately favor
    # "already close enough, don't overshoot", same as the arrival-radius fade
    # in the real cost function - the regression this guards is freezing well
    # short of a target that's comfortably within the reachable range.
    for target_x in (0.6, 0.9, 1.2, 1.4):
        vx1, omega1, vx2, omega2 = _pick_two_stage(np.array([target_x, 0.0, 0.0]))
        assert vx1 > 0.0, f"target at distance {target_x} yields vx1={vx1} (planner would never move)"

def test_reverse_gate_hysteresis_ignores_noise_at_single_threshold():
    # regression: a rosbag capture showed the old single-threshold gate
    # (should_reverse = front_clearance <= 0.30) flipping forward<->reverse
    # every planning cycle when front_clearance hovered right at 0.30 with
    # ESDF/depth noise - each flip swaps a 1e9-cost gap between forward and
    # reverse candidates, so the robot's actual path alternated wildly.
    in_reverse = False
    for clearance in (0.30, 0.29, 0.31, 0.30, 0.29, 0.31):
        in_reverse = reverse_gate_hysteresis(clearance, in_reverse)
        assert in_reverse is False, "noise straddling the old single threshold must not enter reverse"

def test_reverse_gate_hysteresis_enters_and_exits():
    in_reverse = False
    in_reverse = reverse_gate_hysteresis(0.5, in_reverse)
    assert in_reverse is False, "plenty of clearance should stay in forward mode"
    in_reverse = reverse_gate_hysteresis(0.20, in_reverse)
    assert in_reverse is True, "clearance under the enter threshold must trigger reverse"
    # once in reverse, clearance has to recover past the (higher) exit
    # threshold, not just back above the enter threshold, to leave reverse
    in_reverse = reverse_gate_hysteresis(0.30, in_reverse)
    assert in_reverse is True, "clearance between the two thresholds must stay in reverse"
    in_reverse = reverse_gate_hysteresis(0.45, in_reverse)
    assert in_reverse is False, "clearance past the exit threshold should leave reverse"

if __name__ == "__main__":
    test_goal_heading_error()
    test_goal_behind_curves_toward_target()
    test_goal_ahead_still_drives_straight()
    test_goal_abeam_turns_while_driving()
    test_heading_fades_within_arrival_radius()
    test_heading_fade_is_monotonic_in_distance()
    test_run_raycasting_comparison()

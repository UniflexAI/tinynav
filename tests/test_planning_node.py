
import ast
import numpy as np
import time
import sys
import os
from numba import njit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from std_msgs.msg import Header
from math_utils import matrix_to_quat
from planning_node import (run_raycasting_loopy, build_route_fields, route_band_fade,
                           route_heading_penalty, score_trajectories_by_ESDF,
                           footprint_lattice)
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

def test_build_route_fields_no_route():
    path_dist_map, remaining_map, _, has_route = build_route_fields(
        [], (20, 20), np.array([0.0, 0.0]), 0.1,
    )
    assert not has_route
    assert np.all(path_dist_map == 1e3)
    assert np.all(remaining_map == 1e3)

def test_build_route_fields_straight_line():
    origin = np.array([0.0, 0.0])
    resolution = 0.1
    # a straight 4m route along x, at y=2.5
    route_xy = [(0.5, 2.5), (4.5, 2.5)]
    path_dist_map, remaining_map, route_heading_map, has_route = build_route_fields(
        route_xy, (50, 50), origin, resolution,
    )
    assert has_route

    def cell(x, y):
        return int((x - origin[0]) / resolution), int((y - origin[1]) / resolution)

    start_r, start_c = cell(0.5, 2.5)
    end_r, end_c = cell(4.5, 2.5)
    on_route_r, on_route_c = cell(2.5, 2.5)
    off_route_r, off_route_c = cell(2.5, 0.5)

    assert abs(remaining_map[start_r, start_c] - 4.0) < 0.2
    assert abs(remaining_map[end_r, end_c] - 0.0) < 0.2
    assert abs(remaining_map[on_route_r, on_route_c] - 2.0) < 0.2
    # on the line: nearest route cell is itself
    assert path_dist_map[on_route_r, on_route_c] < resolution
    # 2m off the line: distance to the route should reflect that
    assert abs(path_dist_map[off_route_r, off_route_c] - 2.0) < 0.2

def test_score_trajectories_by_esdf_route_terms():
    origin = np.array([0.0, 0.0])
    resolution = 0.1
    route_xy = [(0.5, 2.5), (4.5, 2.5)]
    path_dist_map, remaining_map, route_heading_map, has_route = build_route_fields(
        route_xy, (50, 50), origin, resolution,
    )
    assert has_route

    ESDF_map = np.full((50, 50), 5.0, dtype=np.float32)  # far from every obstacle

    # a trajectory that tracks the route from start to end
    on_route_traj = np.array([[
        [x, 2.5, 0.0, 0.0, 0.0, 0.0, 1.0] for x in np.linspace(0.6, 4.4, 5)
    ]])
    scores, occ_points, path_costs, end_remainings, end_heading_errs = score_trajectories_by_ESDF(
        on_route_traj, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    assert scores[0] == 0.0  # nothing anywhere near safety_radius
    assert path_costs[0] < resolution  # trajectory center never leaves the route
    assert abs(end_remainings[0] - 0.0) < 0.2  # ends at the route's end

    # a trajectory that runs parallel to the route, 2m off to the side
    off_route_traj = np.array([[
        [x, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0] for x in np.linspace(0.6, 4.4, 5)
    ]])
    _, _, off_path_costs, _, _ = score_trajectories_by_ESDF(
        off_route_traj, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    assert abs(off_path_costs[0] - 2.0) < 0.2


def _pose_facing(x, y, heading):
    """A pose7 [x, y, z, qx, qy, qz, qw] whose BODY +Z (this planner's forward) points
    along `heading` in world XY -- the camera convention every heading in
    planning_node uses. The rotation's columns are (right, down, forward), so the
    scoring's own heading_of_pose7 reads `heading` back."""
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[s, 0.0, c],
                  [-c, 0.0, s],
                  [0.0, -1.0, 0.0]])
    return [x, y, 0.0, *matrix_to_quat(R)]


def test_route_heading_is_the_route_direction_not_the_bearing_to_the_goal():
    """At a corner, the route's own direction -- not the bearing to the goal -- is
    what a candidate is measured against, and carrying straight on reads as wrong.

    Both candidates end within centimetres of the route, so `path_dist` and
    `remaining` cannot tell them apart; the heading is the only thing that can, which
    is why the robot drove straight through a right turn on 118 (2026-08-31) while
    nothing measured it.
    """
    origin = np.array([0.0, 0.0])
    resolution = 0.1
    # the route runs +x for 2 m, then turns and runs +y for 2 m
    route_xy = [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5)]
    path_dist_map, remaining_map, route_heading_map, has_route = build_route_fields(
        route_xy, (50, 50), origin, resolution,
    )
    assert has_route

    def heading_at(x, y):
        return route_heading_map[int(x / resolution), int(y / resolution)]

    assert abs(heading_at(1.5, 0.5) - 0.0) < 0.2                 # along +x before it
    assert abs(heading_at(2.5, 1.5) - np.pi / 2) < 0.2           # along +y after it

    ESDF_map = np.full((50, 50), 5.0, dtype=np.float32)
    # from just before the corner: one candidate carries straight on, one turns into it
    straight = np.array([[_pose_facing(x, 0.5, 0.0) for x in np.linspace(2.0, 3.0, 5)]])
    turning = np.array([[_pose_facing(2.5, y, np.pi / 2) for y in np.linspace(0.5, 1.5, 5)]])

    _, _, _, _, straight_err = score_trajectories_by_ESDF(
        straight, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    _, _, _, _, turning_err = score_trajectories_by_ESDF(
        turning, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    assert turning_err[0] < np.deg2rad(20), turning_err[0]
    assert straight_err[0] > np.deg2rad(60), straight_err[0]

def test_the_goal_takes_over_from_the_route_across_the_terminal_band():
    """One encoding of the band. The heading term fades out over it and the terminal
    position term fades in; written twice they would drift, and on arrival the robot
    would be pulled toward two different headings."""
    band = 0.5
    assert route_band_fade(3.0, band) == 1.0        # far out: the route decides
    assert route_band_fade(0.0, band) == 0.0        # arrived: the goal decides
    assert 0.0 < route_band_fade(0.25, band) < 1.0  # and it hands over, not switches


def test_carrying_straight_on_at_a_corner_costs_more_than_turning():
    """The penalty the cost actually adds, not just the field it reads.

    Deleting the term from cost_function leaves the heading test above green -- this
    is the one that goes red, because it asserts the number that changes the choice.
    """
    w, band = 60.0, 0.5
    turning = route_heading_penalty(w, np.deg2rad(5), 3.0, band)
    straight = route_heading_penalty(w, np.deg2rad(85), 3.0, band)
    assert straight > turning
    # ... by enough to outweigh the smoothness term (10 * |d omega|, at most ~13 over
    # the full omega range) that was winning these ties.
    assert straight - turning > 13.0

    # Inside the terminal band the route has run out: the arrival heading takes over,
    # so this term must fade rather than fight it.
    assert route_heading_penalty(w, np.deg2rad(85), 0.0, band) == 0.0
    assert (route_heading_penalty(w, np.deg2rad(85), 0.25, band)
            < 0.6 * route_heading_penalty(w, np.deg2rad(85), 3.0, band))


if __name__ == "__main__":
    # Upstream's goal-heading tests are deliberately absent: they exercise
    # `goal_heading_error` and a Python `cost_function` closure, and this fork scores
    # the end heading inside `score_trajectories_by_ESDF` off the route heading field
    # instead. What replaces them are the route-heading tests below.
    test_run_raycasting_comparison()
    test_build_route_fields_no_route()
    test_route_heading_is_the_route_direction_not_the_bearing_to_the_goal()
    test_carrying_straight_on_at_a_corner_costs_more_than_turning()
    test_the_goal_takes_over_from_the_route_across_the_terminal_band()
    test_build_route_fields_straight_line()
    test_score_trajectories_by_esdf_route_terms()
    print("Route field tests passed.")


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
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
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
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.35, config)   # above the relaxed threshold
    assert build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config,
                              min_span_map=_span_map(config, (1, 1)))[1, 1]


def test_min_span_map_relaxes_steps_above_the_ground_band():
    """Mid-climb the next riser starts above robot_z_bottom + ground_band_m; a relaxed
    cell must still read it as a step, not as a floating obstacle."""
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
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
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.15, config)
    assert np.array_equal(
        build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config),
        build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config,
                           min_span_map=None))


# --- published Path: the stride cmd_vel_control's dt assumption rests on -----

def _publish_path_recorder():
    """A PlanningNode with only what publish_selected_path touches, via __new__ so none of
    __init__'s model/map loading runs."""
    from planning_node import PlanningNode
    node = PlanningNode.__new__(PlanningNode)
    sent = []
    node.path_pub = type('P', (), {'publish': lambda _s, m: sent.append(m)})()
    return node, sent


def _straight_traj(n, dx=0.1):
    traj = np.zeros((n, 7))
    traj[:, 0] = np.arange(n) * dx
    traj[:, 6] = 1.0
    return np.array([traj])


def test_published_path_keeps_every_tenth_pose():
    # cmd_vel_control derives speed and turn rate from this Path as
    # planner_dt * path_pose_stride * step_idx; publishing at another stride scales
    # both by that ratio, so the number is part of the interface, not a display choice.
    from planning_node import PlanningNode
    node, sent = _publish_path_recorder()
    trajs = _straight_traj(31)
    node.publish_selected_path(trajs, [0], Header())
    assert len(sent) == 1
    assert PlanningNode.PATH_POSE_STRIDE == 10
    assert len(sent[0].poses) == 4  # 0, 10, 20, 30
    xs = [p.pose.position.x for p in sent[0].poses]
    assert np.allclose(xs, [0.0, 1.0, 2.0, 3.0])


def test_published_path_preserves_direction_of_travel():
    # What the all-trajectories-collide fallback relies on: cmd_vel_control reads
    # reverse off the path pointing backwards, so the sign has to survive publishing.
    node, sent = _publish_path_recorder()
    node.publish_selected_path(_straight_traj(21, dx=-0.1), [0], Header())
    xs = [p.pose.position.x for p in sent[0].poses]
    assert xs[1] < xs[0]


def _planning_source():
    path = os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core',
                        'planning_node.py')
    with open(path) as fh:
        return ast.parse(fh.read())


def test_no_statement_sits_after_a_return():
    """Nothing in this file may be unreachable.

    Not style: a hand-merge once landed PlanningNode's route setup -- the TF buffer,
    the /mapping/global_plan subscription and `_global_route_map_xy` -- after the
    `return` in `_open_target_speed`. It parsed, it imported, the node started, and
    then the first sync_callback died with AttributeError because the attribute had
    never been created. The planner was gone and nothing published a trajectory.
    """
    offenders = []
    for node in ast.walk(_planning_source()):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for i, stmt in enumerate(node.body[:-1]):
            if isinstance(stmt, (ast.Return, ast.Raise)):
                offenders.append(f'{node.name}: line {node.body[i + 1].lineno} is '
                                 f'unreachable after {type(stmt).__name__} on line '
                                 f'{stmt.lineno}')
    assert not offenders, offenders


def test_every_attribute_the_route_reads_is_built_in_init():
    """`_route_in_world` runs from `sync_callback`, which fires as soon as images
    arrive -- long before anything else has had a chance to create its state. So
    what it reads has to exist from construction, not from whichever method
    happened to be edited alongside it.
    """
    tree = _planning_source()
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'PlanningNode')
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    def self_attrs_read(fn):
        return {n.attr for n in ast.walk(fn)
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
                and isinstance(n.value, ast.Name) and n.value.id == 'self'}

    def self_attrs_set(fn):
        return {t.attr for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == 'self'}

    built = self_attrs_set(fns['__init__'])
    methods = set(fns)
    missing = sorted(a for a in self_attrs_read(fns['_route_in_world'])
                     if a not in built and a not in methods)
    assert not missing, f'_route_in_world reads what __init__ never builds: {missing}'


# --- footprint sampling ---------------------------------------------------------
# A b2 as the planner sees it: 0.80 x 0.30 m about the control centre.
_B2 = dict(front_len=0.4, rear_len=0.4, half_w=0.15, safety_radius=0.1)


def _flat_fields(shape):
    """Route maps that rank nothing, so only the clearance score moves."""
    return (np.zeros(shape), np.zeros(shape), np.zeros(shape))


def _esdf_with_obstacle(shape, resolution, origin, obst_xy):
    """A true Euclidean distance field to a single obstacle cell."""
    xs = origin[0] + resolution * np.arange(shape[0])
    ys = origin[1] + resolution * np.arange(shape[1])
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    return np.sqrt((gx - obst_xy[0]) ** 2 + (gy - obst_xy[1]) ** 2)


def _score_standing_at(pose_xy, obst_xy):
    """Clearance score for a robot standing still at `pose_xy`, facing +x."""
    shape, resolution, origin = (120, 120), 0.05, np.array([-1.0, -1.0])
    ESDF = _esdf_with_obstacle(shape, resolution, origin, obst_xy)
    path_dist, remaining, route_heading = _flat_fields(shape)
    # facing +x with the body-+z-forward convention this file uses
    q = matrix_to_quat(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    traj = np.array([[[pose_xy[0], pose_xy[1], 0.0, q[0], q[1], q[2], q[3]]] * 4])
    scores, _, _, _, _ = score_trajectories_by_ESDF(
        traj, ESDF, path_dist, remaining, route_heading, origin, resolution,
        _B2['safety_radius'], _B2['front_len'], _B2['rear_len'], _B2['half_w'])
    return scores[0]


def test_an_obstacle_against_the_nose_is_not_open_space():
    """The bug: the middle of the front edge sat 0.15 m from either front corner,
    so with only centre+corners sampled an obstacle touching the robot's nose read
    min ESDF 0.15 m -- above safety_radius, scored 0.0, ranked as open space."""
    nose = _score_standing_at((0.0, 0.0), (0.4, 0.0))
    assert nose > 0.0, 'an obstacle at the front-edge midpoint scored as open space'

    side = _score_standing_at((0.0, 0.0), (0.0, 0.15))
    assert side > 0.0, 'an obstacle at the side-edge midpoint scored as open space'


def test_open_space_still_scores_zero():
    """The other half of the same claim: a denser lattice must not start finding
    obstacles where there are none, or every trajectory pays and the robot stops."""
    assert _score_standing_at((0.0, 0.0), (5.0, 5.0)) == 0.0


def test_no_point_of_the_footprint_is_further_than_safety_radius_from_a_sample():
    """The property the pitch is derived from, asserted against the lattice the
    scorer actually uses: every point of the footprint must sit within
    safety_radius of some sample, or an obstacle can hide between them and read
    as clearance."""
    fl, rl, hw = _B2['front_len'], _B2['rear_len'], _B2['half_w']
    off_fwd, off_lat = footprint_lattice(fl, rl, hw, _B2['safety_radius'])
    samples = list(zip(off_fwd, off_lat))
    assert (0.0, 0.0) in samples, 'the centre must stay sample 0 -- the route reads it'

    n = 121
    worst = 0.0
    for i in range(n):
        for j in range(n):
            x = -rl + (fl + rl) * i / (n - 1)
            y = -hw + 2 * hw * j / (n - 1)
            worst = max(worst, min(np.hypot(x - sx, y - sy) for sx, sy in samples))
    assert worst <= _B2['safety_radius'] + 1e-9, (
        f'a point of the footprint sits {worst:.3f} m from the nearest sample, '
        f'further than safety_radius {_B2["safety_radius"]} m -- an obstacle can '
        'hide there and read as clearance')


def test_the_lattice_covers_every_robot_the_repo_ships():
    """The pitch is derived, so this must hold for any footprint, not just a b2."""
    for fl, rl, hw in ((0.4, 0.4, 0.15), (0.25, 0.35, 0.15), (0.3, 0.3, 0.3)):
        off_fwd, off_lat = footprint_lattice(fl, rl, hw, 0.1)
        samples = list(zip(off_fwd, off_lat))
        worst = 0.0
        for i in range(61):
            for j in range(61):
                x = -rl + (fl + rl) * i / 60
                y = -hw + 2 * hw * j / 60
                worst = max(worst, min(np.hypot(x - sx, y - sy) for sx, sy in samples))
        assert worst <= 0.1 + 1e-9, (
            f'footprint {fl+rl:.2f}x{2*hw:.2f} leaves a {worst:.3f} m gap')


# --- the all-collision escape ---------------------------------------------------
class _EscapeNode:
    """Just the two methods the escape decision is built from, on a real b2
    footprint. `_footprint_hits` is the shipped one -- bound off PlanningNode so a
    change to the lattice or to the sampling reaches this test."""

    def __init__(self, obstacle_xy=None, resolution=0.05, origin=(-2.0, -2.0)):
        self.resolution = resolution
        self.origin = np.array(origin)
        n = 120
        self.mask = np.zeros((n, n), dtype=bool)
        if obstacle_xy is not None:
            xi = int((obstacle_xy[0] - origin[0]) / resolution)
            yi = int((obstacle_xy[1] - origin[1]) / resolution)
            self.mask[xi, yi] = True

    def camera_to_robot_center(self, T):
        return np.array([0.0, 0.0, 0.0])


def _hits_at(obstacle_xy):
    """The shipped `_footprint_hits`, for a robot at the origin facing +x."""
    from planning_node import PlanningNode
    node = _EscapeNode(obstacle_xy)
    # body +z forward, so the rotation puts world +x under the body's forward axis
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return PlanningNode._footprint_hits(node, T, node.mask)


def test_a_cell_under_the_body_is_reported_as_a_body_hit():
    """It is what says the standing-still row itself collides: a row that never
    leaves the current pose can only be inf if a footprint sample is already on an
    obstacle cell. Sampled through the shipped lattice, so a cell has to land on a
    sample -- exactly the condition the scorer uses."""
    off_fwd, off_lat = footprint_lattice(0.25, 0.35, 0.15, 0.1)   # the go2 default
    on_sample = (float(off_fwd[3]), float(off_lat[3]))
    hits, n = _hits_at(on_sample)
    assert hits, f'a cell on sample {on_sample} was not seen'
    assert n == len(off_fwd)


def test_a_cell_beyond_the_footprint_is_not_a_body_hit():
    """The paired normal path: standing on clear ground reports no body hits, so
    the all-collision branch is not where an open-ground stall lands."""
    hits, _ = _hits_at((1.50, 0.0))        # 1.1 m past the nose
    assert hits == [], f'a cell 1.5 m away was read as under the robot: {hits}'


def test_the_report_names_where_the_hits_are():
    """Scattered singles are odometry-drift phantoms; a blob is a real obstacle.
    The offsets are the only way to tell, so they have to be in the line."""
    from planning_node import PlanningNode
    off_fwd, off_lat = footprint_lattice(0.25, 0.35, 0.15, 0.1)
    hits, n = _hits_at((float(off_fwd[3]), float(off_lat[3])))
    line = PlanningNode._hits_report(hits, n)
    assert 'body_hits=%d/%d' % (len(hits), n) in line
    assert '(' in line and ',' in line, f'no offsets in the report: {line}'
    assert PlanningNode._hits_report([], n) == 'body_hits=0/%d' % n


def _sync_callback_src():
    src = open(os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core',
                            'planning_node.py')).read()
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)
                and n.name == 'sync_callback')


def test_the_reverse_family_is_armed_by_having_no_way_forward():
    """The freeze on 122 sat at front_clearance 0.60-0.75 with every forward
    trajectory in collision, and the clearance proxy answered `should_reverse=False`
    -- so standing still, which is neither colliding nor gated, stayed the cost
    minimum for 21s. Whether there IS a way forward is measured, not proxied."""
    fn = _sync_callback_src()
    assign = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                  and any(getattr(t, 'id', None) == 'should_reverse' for t in n.targets))
    names = {n.id for n in ast.walk(assign.value) if isinstance(n, ast.Name)}
    assert 'n_fwd_ok' in names, (
        'should_reverse is back to reading only the forward corridor: '
        f'{sorted(names)}')
    assert 'front_clearance' in names, (
        'the close-obstacle case was dropped along with the proxy')


def test_the_all_collision_branch_publishes_nothing():
    """Reaching it means the standing-still row is inf too, so a footprint sample
    is already on an obstacle cell and every trajectory is inf whichever way it
    points. There is no direction in the scores, so an escape there would be
    blind -- and a reverse row published from it was dead code."""
    fn = _sync_callback_src()
    allcoll = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                   and isinstance(n.test, ast.Call)
                   and getattr(n.test.func, 'id', None) == 'all')
    calls = {getattr(c.func, 'attr', None) for c in ast.walk(allcoll)
             if isinstance(c, ast.Call)}
    assert 'publish_selected_path' not in calls, (
        'the all-collision branch publishes a path again -- from scores that '
        'cannot tell one direction from another')

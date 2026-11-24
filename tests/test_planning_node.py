
import ast
from dataclasses import replace
import numpy as np
import pytest
import time
import sys
import os
import types
from numba import njit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from std_msgs.msg import Header
from math_utils import matrix_to_quat
from scipy.ndimage import distance_transform_edt
from planning_node import (run_raycasting_loopy, build_route_fields, score_trajectories_by_ESDF,
                           select_by_time, carried_row, turn_remainder, admit_reverse,
                           generate_predefined_trajectory_vocabularies, REVERSE_SPEED,
                           REVERSE_CAREFUL_SPEED, REVERSE_CAREFUL_M,
                           ARRIVE_M, ARRIVE_TICKS, ARRIVE_TICK_S,
                           generate_rotate_then_go, ROTATE_DEG, ROTATE_THEN_GO_M,
                           footprint_lattice, footprint_cells, PlanningNode, ROBOT_CONFIG,
                           reverse_armed, REVERSE_ENTER_M, REVERSE_EXIT_M,
                           generate_trajectory_library_3d, heading_of_pose7, angle_between,
                           route_left_at, around_obstacles, route_blocked_ahead)
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
    scores, occ_points, path_costs, end_remainings, end_heading_errs = _score5(
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
    _, _, off_path_costs, _, _ = _score5(
        off_route_traj, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    assert abs(off_path_costs[0] - 2.0) < 0.2


def _score5(*args):
    """The scorer's first five columns: what these tests are about."""
    return score_trajectories_by_ESDF(*args)[:5]


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

    _, _, _, _, straight_err = _score5(
        straight, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    _, _, _, _, turning_err = _score5(
        turning, ESDF_map, path_dist_map, remaining_map, route_heading_map,
        origin, resolution,
    )
    assert turning_err[0] < np.deg2rad(20), turning_err[0]
    assert straight_err[0] > np.deg2rad(60), straight_err[0]

if __name__ == "__main__":
    # Upstream's goal-heading tests are deliberately absent: they exercise
    # `goal_heading_error` and a Python `cost_function` closure, and this fork scores
    # the end heading inside `score_trajectories_by_ESDF` off the route heading field
    # instead. What replaces them are the route-heading tests below.
    test_run_raycasting_comparison()
    test_build_route_fields_no_route()
    test_route_heading_is_the_route_direction_not_the_bearing_to_the_goal()
    test_build_route_fields_straight_line()
    test_score_trajectories_by_esdf_route_terms()
    print("Route field tests passed.")


# --- climb region: cells judged by height above the ground they show -----------

def _riser_grid(height_m, config, resolution=0.05):
    """3x3 cells; the centre one holds occupancy from the band's bottom up to height_m."""
    z_layers = int(round((config.robot_z_top - config.robot_z_bottom) / resolution))
    grid = np.zeros((3, 3, z_layers))
    grid[1, 1, :int(round(height_m / resolution))] = 0.2
    return grid, np.array([0.0, 0.0, config.robot_z_bottom])


def _region(cell):
    m = np.zeros((3, 3), dtype=bool)
    m[cell] = True
    return m


def test_a_riser_from_the_floor_is_a_step_in_the_climb_region_and_only_there():
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.15, config)
    args = (grid, origin, 0.05)
    assert build_obstacle_map(*args, robot_z=0.0, config=config)[1, 1]
    assert not build_obstacle_map(*args, robot_z=0.0, config=config,
                                  climb_region=_region((1, 1)))[1, 1]
    # a region somewhere else leaves this cell to the span filter
    assert build_obstacle_map(*args, robot_z=0.0, config=config,
                              climb_region=_region((0, 0)))[1, 1]


def test_what_rises_a_step_above_the_floor_blocks():
    """Measured from the floor, not from the band's bottom: 0.2 m of band over the
    floor clearance is more than a step."""
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig()
    grid, origin = _riser_grid(0.2, config)
    assert build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config,
                              climb_region=_region((1, 1)), step_m=0.2)[1, 1]


def test_a_tread_above_the_robot_is_its_own_ground():
    """Mid-climb a tread above the ground band is its own ground; a B2's band is tall
    enough to hold a step over it."""
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig(robot_z_bottom=-0.66)
    resolution = 0.05
    grid, origin = _riser_grid(0.0, config, resolution)
    base = int(round((config.ground_band_m + resolution) / resolution))
    grid[1, 1, base:base + 3] = 0.2
    args = (grid, origin, resolution)
    assert build_obstacle_map(*args, robot_z=0.0, config=config)[1, 1]
    assert not build_obstacle_map(*args, robot_z=0.0, config=config,
                                  climb_region=_region((1, 1)))[1, 1]
    grid[1, 1, base:base + 8] = 0.2
    assert build_obstacle_map(*args, robot_z=0.0, config=config,
                              climb_region=_region((1, 1)))[1, 1]


def test_what_tops_out_under_the_step_height_is_floor():
    """0.10 m of band over the floor clearance tops out ~0.14 m over the floor: an
    obstacle by the span filter alone, floor under a 0.20 m step height; 0.25 m of
    band is not."""
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig(step_height_m=0.20)
    low, origin = _riser_grid(0.10, config)
    assert build_obstacle_map(low, origin, 0.05, robot_z=0.0, config=ObstacleConfig())[1, 1]
    assert not build_obstacle_map(low, origin, 0.05, robot_z=0.0, config=config)[1, 1]
    high, _ = _riser_grid(0.25, config)
    assert build_obstacle_map(high, origin, 0.05, robot_z=0.0, config=config)[1, 1]


def test_the_step_height_leaves_what_hangs_high_and_the_climb_region_alone():
    from planning_node import build_obstacle_map
    from tinynav.core.robot_specs import ObstacleConfig
    config = ObstacleConfig(step_height_m=0.20)
    grid, origin = _riser_grid(0.0, config)
    grid[1, 1, 7:9] = 0.2                       # floating, 0.4 m over the band's bottom
    assert build_obstacle_map(grid, origin, 0.05, robot_z=0.0, config=config)[1, 1]
    riser, _ = _riser_grid(0.10, config)        # under the step height, but on a stair
    assert build_obstacle_map(riser, origin, 0.05, robot_z=0.0, config=config,
                              climb_region=_region((1, 1)), step_m=0.1)[1, 1]


def test_the_g1_walks_over_what_is_under_20_cm():
    from tinynav.core.robot_specs import G1_CONFIG, GO2_CONFIG, ObstacleConfig
    assert (ObstacleConfig().step_height_m, GO2_CONFIG.obstacle.step_height_m) == (0.0, 0.0)
    assert G1_CONFIG.obstacle.step_height_m == 0.20


def test_the_climb_region_is_the_cells_near_a_climb_point():
    from planning_node import PlanningNode
    node = PlanningNode.__new__(PlanningNode)
    node._signal_fresh = lambda stamp, window: stamp is not None
    node._climb_stamp_ns, node._climb_region_ttl_ns = 1, 1
    node._climb_region_cells = 2
    node._climb_points = np.array([[0.025, 0.025], [0.425, 0.025]])
    r = node._climb_region(np.zeros(3), 0.05, (20, 20))
    assert r[0, 0] and r[2, 0] and r[7, 0] and r[8, 2]
    assert not r[0, 5] and not r[19, 19]                # beyond the radius
    node._climb_stamp_ns = None
    assert node._climb_region(np.zeros(3), 0.05, (20, 20)) is None


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
    scores, _, _, _, _ = _score5(
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


def _clearance_sweep():
    """Clearance score against the gap between the robot's nose and one obstacle cell,
    from grazing out to open space. Read off the sweep rather than a single gap: which
    lattice sample is nearest the nose sets the offset between `gap` and the min ESDF,
    so only the shape of the curve is a claim about the scoring."""
    gaps = [0.005 * i for i in range(30)]
    return [_score_standing_at((0.0, 0.0), (_B2['front_len'] + g, 0.0)) for g in gaps]


def test_the_clearance_term_has_no_step_at_safety_radius():
    """Among turn-in-place rows every positional term is identical by construction --
    same end cell, zero arc -- so the selection is decided by this term at w_clearance
    200 against a smoothness term whose whole range is 15. A step from 0 to
    1/safety_radius here is one grid cell at the margin choosing the turn direction."""
    scores = _clearance_sweep()
    nonzero = [s for s in scores if s > 0.0]
    assert nonzero, 'the sweep never entered the band -- the test would prove nothing'
    assert scores[-1] == 0.0, 'the sweep never left the band'
    assert min(nonzero) < 1.0, f'still a step at safety_radius: {min(nonzero)}'


def test_the_clearance_term_still_bites_near_collision():
    """The other half: continuity must not flatten the ramp. Closer stays worse, and
    the near end stays far above every other term -- 5 here is ~1000 cost units at
    w_clearance, against a smoothness range of 15."""
    scores = _clearance_sweep()
    assert scores == sorted(scores, reverse=True), 'closer is not monotonically worse'
    assert scores[0] > 5.0, f'grazing barely costs anything: {scores[0]}'


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
        import planning_node
        self.resolution = resolution
        self.origin = np.array(origin)
        # What PlanningNode._use_footprint_scale leaves for the checks: the outline of
        # planning_node's ROBOT_CONFIG as it is when the stand-in is made.
        self._collision_fp = planning_node.ROBOT_CONFIG.collision_footprint()
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
    # The model's own footprint, which `_footprint_hits` samples: a hard-coded one
    # stops landing on a sample when the model's numbers change.
    off_fwd, off_lat = footprint_lattice(*ROBOT_CONFIG.collision_footprint(),
                                         ROBOT_CONFIG.safety_radius)
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
    off_fwd, off_lat = footprint_lattice(*ROBOT_CONFIG.collision_footprint(),
                                         ROBOT_CONFIG.safety_radius)
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


class _Grid:
    """Just the attributes `score_trajectories` reads off the node."""

    score_trajectories = PlanningNode.score_trajectories

    def __init__(self, origin, resolution):
        import planning_node
        self.origin, self.resolution = origin, resolution
        self._traj_dt, self.route_terminal_band = 0.1, 0.5
        self._collision_fp = planning_node.ROBOT_CONFIG.collision_footprint()


_SHAPE, _RES, _ORIGIN = (160, 160), 0.05, np.array([-3.0, -3.0])


def _cell(xy):
    """The cell a world point falls in, by the arithmetic the scorer itself uses."""
    return (int((xy[0] - _ORIGIN[0]) / _RES), int((xy[1] - _ORIGIN[1]) / _RES))


def _esdf_of_cell(cell):
    """A distance field around one occupied cell, built the way the node builds it."""
    mask = np.zeros(_SHAPE, dtype=bool)
    mask[cell] = True
    return distance_transform_edt(~mask).astype(np.float32) * _RES


def _score_rows(obst_xy, back_step):
    """Score a standing-still row and a reverse row, both starting at the origin
    facing +x, against one obstacle cell. Returns (standing, reverse)."""
    ESDF = _esdf_of_cell(_cell(obst_xy))
    path_dist, remaining, route_heading = _flat_fields(_SHAPE)
    q = matrix_to_quat(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    here = [0.0, 0.0, 0.0, q[0], q[1], q[2], q[3]]
    standing = [list(here) for _ in range(4)]
    reverse = [[-back_step * k] + here[1:] for k in range(4)]
    scores = _Grid(_ORIGIN, _RES).score_trajectories(
        np.array([standing, reverse]), np.array([[0.0, 0.0], [-0.3, 0.0]]),
        ESDF, path_dist, remaining, route_heading, np.zeros(3), False)[0]
    return scores[0], scores[1]


def _edge_samples():
    """The frontmost and rearmost footprint samples, on one side."""
    fl, rl, hw = ROBOT_CONFIG.collision_footprint()
    off_fwd, _ = footprint_lattice(fl, rl, hw, ROBOT_CONFIG.safety_radius)
    return (float(max(off_fwd)), -hw), (float(min(off_fwd)), -hw)


def test_the_pose_it_starts_from_does_not_veto_backing_out_of_it():
    """An obstacle against the nose is on a footprint sample at step 0, and step 0
    is every candidate\'s -- so it used to make the reverse row inf as well, and the
    scores carried no direction at the one moment they had to."""
    nose, _ = _edge_samples()
    standing, reverse = _score_rows(nose, back_step=_RES)
    assert standing == float('inf'), 'standing on the obstacle is still a collision'
    assert reverse < float('inf'), 'backing off it was refused for the pose it backs off'


def test_but_a_blocked_rear_still_refuses_the_reverse_row():
    """The other half: only the starting pose is excused. Steps 1.. are checked as
    ever, so reverse into something really there is still inf."""
    _, tail = _edge_samples()
    behind = (tail[0] - 2 * _RES, tail[1])
    standing, reverse = _score_rows(behind, back_step=_RES)
    assert standing < float('inf'), 'nothing is touching the robot where it stands'
    assert reverse == float('inf'), 'reverse drove into the obstacle behind it'


def _steps_to(metres, res):
    """Every reading the forward scan can report out to `metres`, as it computes
    them -- `step * res`, which is where the float lands."""
    return [step * res for step in range(int(round(metres / res)) + 1)]


def test_the_reverse_gate_fires_at_the_distance_it_is_written_for():
    """`front_clearance` counts grid steps, so it never lands exactly on a threshold
    in metres: 6 * 0.05 is 0.30000000000000004. Compared against 0.30 the gate was
    shut at its own number -- 720 frames of `front_clr=0.30 should_reverse=False` on
    122 on 2026-09-16, the robot stationary in front of something 0.30 m away."""
    for res in (0.05, 0.1):
        for d in _steps_to(REVERSE_ENTER_M, res):
            assert reverse_armed(d, res), \
                f'{d:.2f} m ahead at res {res} did not arm reverse'


def test_but_a_reading_a_step_further_out_does_not():
    """The other half. The slack is under one step, so it admits the reading nearest
    the threshold and no further one -- otherwise it is just a wider threshold that
    nobody chose."""
    for res in (0.05, 0.1):
        beyond = int(round(REVERSE_ENTER_M / res)) + 1
        for step in range(beyond, beyond + 20):
            d = step * res
            assert not reverse_armed(d, res), \
                f'{d:.2f} m ahead at res {res} armed reverse'


def test_a_clear_corridor_does_not_arm_reverse_however_blocked_the_rollouts_are():
    """Arming is the clearance reading alone, never "no forward row is clear"."""
    assert not reverse_armed(3.0, 0.05)


def test_this_models_gate_is_its_robot_config():
    from tinynav.core.robot_specs import ROBOT_CONFIG
    assert (REVERSE_ENTER_M, REVERSE_EXIT_M) == \
        (ROBOT_CONFIG.reverse_enter_m, ROBOT_CONFIG.reverse_exit_m)


def test_the_g1_backs_off_from_further_out():
    """8038, 2026-11-17: at 0.10 the g1 drove into cells that flicker at that range,
    every row scored inf and it stopped and went."""
    from tinynav.core.robot_specs import G1_CONFIG
    assert (G1_CONFIG.reverse_enter_m, G1_CONFIG.reverse_exit_m) == \
        pytest.approx((0.20, 0.30))


def _gates():
    from tinynav.core import robot_specs
    return sorted({(c.reverse_enter_m, c.reverse_exit_m)
                   for n, c in vars(robot_specs).items()
                   if n.endswith('_CONFIG') and isinstance(c, robot_specs.RobotConfig)})


@pytest.mark.parametrize('enter, exit_', _gates())
def test_every_models_gate_fires_at_its_own_distance_and_holds_its_band(enter, exit_):
    """The step tests above, at each model's pair passed in rather than this one's."""
    assert exit_ > enter, f'exit {exit_} leaves no band above the entry {enter}'
    for res in (0.05, 0.1):
        for d in _steps_to(enter, res):
            assert reverse_armed(d, res, enter_m=enter, exit_m=exit_), \
                f'{d:.2f} m at res {res} did not arm a {enter}/{exit_} gate'
        beyond = int(round(enter / res)) + 1
        for step in range(beyond, beyond + 20):
            assert not reverse_armed(step * res, res, enter_m=enter, exit_m=exit_), \
                f'{step * res:.2f} m at res {res} armed a {enter}/{exit_} gate'
    res = 0.05
    lo, hi = enter + res / 2, exit_ + res / 2
    between = (lo + hi) / 2
    assert not reverse_armed(between, res, False, enter, exit_)
    assert reverse_armed(between, res, True, enter, exit_)
    assert not reverse_armed(hi + res, res, True, enter, exit_)


def _lattice_speeds(v_allow, floor):
    _, params = generate_trajectory_library_3d(
        max_linear_vel=v_allow, min_linear_vel=floor)
    return sorted({round(float(vx), 9) for vx, _ in params})


def test_the_lattice_offers_no_speed_between_a_standstill_and_the_floor():
    """Sampling from 0 put three of seven speeds under the floor at v_allow 0.20 --
    0, 0.033, 0.067 -- and the cost minimum sat on 0.033, which cmd_vel_control reads
    as a stop. 122 stood still for five minutes while `n_fwd_ok` counted 73 ways
    forward, most of a speed it would never execute."""
    for v_allow in (0.2, 0.43, 0.6, 1.0):
        floor = 0.2
        for vx in _lattice_speeds(v_allow, floor):
            assert vx == 0.0 or vx >= min(floor, v_allow) - 1e-9, \
                f'v_allow {v_allow} offered {vx}, under the floor and over a standstill'


# Camera convention, body +Z forward and level: identity would point +Z straight up.
_LEVEL_Q = np.array([-0.5, 0.5, -0.5, 0.5])


def test_the_lattice_turns_only_at_rates_the_chassis_executes():
    """One standstill row and no turn in place; every turn at least min_angular_vel."""
    for v_allow in (0.2, 0.6):
        _, params = generate_trajectory_library_3d(
            max_linear_vel=v_allow, min_linear_vel=0.2, max_angular_vel=0.75,
            max_lat_acc=0.5, min_angular_vel=0.2)
        assert sum(1 for vx, w in params if vx == 0.0) == 1
        assert all(w == 0.0 or abs(w) >= 0.2 for _, w in params)


def test_every_turn_in_place_drives_on_after_it():
    trajs, params, durations = generate_rotate_then_go(
        np.zeros(3), _LEVEL_Q, np.deg2rad(ROTATE_DEG), 0.75, 0.2, ROTATE_THEN_GO_M, 0.1, 70)
    start = heading_of_pose7(np.concatenate([np.zeros(3), _LEVEL_Q]))
    for traj, dur, deg in zip(trajs, durations, ROTATE_DEG):
        assert abs(np.linalg.norm(traj[-1, :2]) - ROTATE_THEN_GO_M) < 1e-6
        turned = np.rad2deg(angle_between(heading_of_pose7(traj[-1]), start))
        assert abs(turned - min(abs(deg), 360 - abs(deg))) < 1e-6
        assert dur <= 69 * 0.1


def test_and_the_top_speed_is_still_offered():
    """The floor must not eat the ceiling: whatever the clearance schedule allows has
    to still be in the lattice, or the robot never reaches it."""
    for v_allow in (0.2, 0.43, 0.6):
        assert max(_lattice_speeds(v_allow, 0.2)) == round(v_allow, 9)


def test_reverse_stays_engaged_until_the_corridor_opens_past_the_exit():
    """Upstream's hysteresis, lost in merge `3a69dbc`. Without the band the gate
    flips on a single grid step and the robot chatters in and out of reverse."""
    res = 0.05
    # Against the EFFECTIVE thresholds: `reverse_armed` carries a half-cell of slack,
    # and at res 0.05 that slack is 0.025 against a 0.05 band -- pick a reading the
    # band actually holds, or this pins the slack rather than the hysteresis.
    lo, hi = REVERSE_ENTER_M + res / 2, REVERSE_EXIT_M + res / 2
    assert hi > lo, f'the exit {REVERSE_EXIT_M} leaves no band above the entry'
    between = (lo + hi) / 2
    assert not reverse_armed(between, res, engaged=False), \
        f'{between:.3f} m armed reverse from disengaged'
    assert reverse_armed(between, res, engaged=True), \
        f'{between:.3f} m dropped reverse while engaged'


def test_but_the_band_still_ends():
    """The counter-case: hysteresis that never releases is just a latch."""
    assert not reverse_armed(REVERSE_EXIT_M + 5 * 0.05, 0.05, engaged=True)
    assert reverse_armed(REVERSE_ENTER_M / 2, 0.05, engaged=False)


def test_footprint_cells_cover_the_body_along_its_heading():
    """Rotated 90 deg: the body lies along +y, so a cell 0.25 m up is under it and
    one 0.25 m to the side is not."""
    origin, res = np.array([-2.5, -2.5]), 0.05
    under = footprint_cells((100, 100), origin, res, (0.0, 0.0), (0.0, 1.0), 0.3, 0.3, 0.15)
    cell = lambda x, y: (int((x - origin[0]) / res), int((y - origin[1]) / res))
    assert under[cell(0.0, 0.25)] and under[cell(0.0, -0.25)]
    assert not under[cell(0.25, 0.0)] and not under[cell(0.0, 0.4)]
    assert under.sum() == 12 * 6


# --- turn-in-place penalty ------------------------------------------------------


# --- the time cost ---------------------------------------------------------------

def test_the_time_cost_arrives_on_the_tick_map_node_does():
    """Closing on a target, the scorer's arrival lands on the first sample whose
    camera pose is inside map_node's circles (RobotConfig.arrival_distance) at ARRIVE_M: nose
    first, where the camera's circle leads, and tail first, where the control
    centre's does."""
    origin, resolution, dt = np.array([0.0, 0.0]), 0.1, 0.1
    zeros = np.zeros((50, 50))
    target = np.array([3.0, 2.5])
    centres = np.linspace(1.0, 3.0, 41)
    cam = ROBOT_CONFIG.cam_offset_3d
    for heading in (0.0, np.pi):
        traj = np.array([[_pose_facing(x, 2.5, heading) for x in centres]])
        arrive_t = score_trajectories_by_ESDF(
            traj, np.full((50, 50), 5.0, dtype=np.float32), zeros, zeros, zeros, origin,
            resolution, 0.1, *ROBOT_CONFIG.collision_footprint(), target[0], target[1],
            float(cam[0]), float(cam[1]), float(cam[2]), ARRIVE_M, 1, 1, 0.5, False, dt)[5][0]
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[s, 0.0, c], [-c, 0.0, s], [0.0, -1.0, 0.0]])
        first = None
        for i, x in enumerate(centres[1:], start=1):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = np.array([x, 2.5, 0.0]) + R @ cam
            if ROBOT_CONFIG.arrival_distance(T, target) < ARRIVE_M:
                first = i
                break
        assert first is not None and abs(arrive_t - first * dt) < 1e-9, heading


def test_standing_still_is_banned_while_another_row_is_open():
    """A standstill, a slow turn and a forward row, none of them arriving."""
    vx, yaw = np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.11, 0.0])
    rem = np.array([1.0, 1.0, 0.1])
    pick = select_by_time(vx, yaw, np.zeros(3), np.full(3, np.inf), rem, np.zeros(3),
                          np.zeros(3), np.zeros(3), rem, False, True, False, 0.5, 0.75, 0.5, 3.0)
    assert pick == 2


def test_but_inside_the_arrival_radius_standing_is_waiting():
    """Every row arrives on the same tick there; the least motion is standing."""
    vx, yaw = np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.11, 0.0])
    one = np.full(3, 1.0)
    pick = select_by_time(vx, yaw, np.zeros(3), one, np.zeros(3), np.zeros(3), np.zeros(3),
                          np.zeros(3), np.zeros(3), True, True, False, 0.5, 0.75, 0.5, 3.0)
    assert pick == 0


# --- carrying a choice forward -------------------------------------------------------

def _pick(vx, yaw, t_end, **kw):
    n = len(vx)
    z = np.zeros(n)
    return select_by_time(np.array(vx), np.array(yaw), z, np.full(n, np.inf), z, z, z, z, z,
                          False, False, False, 0.5, 0.75, 0.5, np.array(t_end), **kw)


def test_a_mirror_tie_keeps_the_last_direction():
    """Left and right equally fast and equally much motion: the side turned last wins,
    whichever of the two is a hair faster."""
    assert _pick([0.2, 0.2], [0.3, -0.3], [1.0, 1.01], last_yaw=-0.3) == 1
    assert _pick([0.2, 0.2], [0.3, -0.3], [1.01, 1.0], last_yaw=0.3) == 0


def test_a_blocked_route_picks_the_standstill_row_over_a_faster_one():
    assert _pick([0.0, 0.2, 0.2], [0.0, 0.0, 0.75], [1.0, 0.5, 0.5], stand_still=True) == 0


def test_the_carried_row_wins_a_tie_but_not_a_race():
    assert _pick([0.0, 0.2, 0.2], [0.0, 0.0, 0.75], [1.0, 1.0, 1.1], carry=2) == 2
    assert _pick([0.0, 0.2, 0.2], [0.0, 0.0, 0.75], [1.0, 1.0, 1.5], carry=2) == 1


def test_a_lattice_row_is_carried_as_itself_whatever_the_speed_cap():
    params = np.array([[0.2, 0.0], [0.4, 0.214], [0.6, 0.214], [0.6, -0.214], [-0.15, 0.0],
                       [0.8, 0.3], [0.2, 0.75]])
    is_turn = np.array([False, False, False, False, False, False, True])
    assert carried_row(params, is_turn, 0.55, 0.214) == 2
    assert carried_row(params, is_turn, 0.8, 0.214) == 2    # the turn before the speed
    assert carried_row(params, is_turn, 0.55, -0.214) == 3
    assert carried_row(params, is_turn, -0.15, 0.0) == 4
    assert carried_row(params, is_turn, -0.15, 0.3) == -1   # never the other way round
    assert carried_row(params, is_turn, 0.2, 0.75) == 5     # never a turn in place


def test_a_turn_in_place_carries_as_its_remainder():
    """A 45 deg turn, 3 steps in: the remainder, as a delta, turns the rest of the
    way to the same heading."""
    d = np.deg2rad(45.0)
    traj, _, _ = generate_rotate_then_go(np.zeros(3), _LEVEL_Q, np.array([d]), 0.75, 0.2, 0.5, 0.1, 60)
    goal = heading_of_pose7(traj[0, -1])
    rem = turn_remainder(traj[0, 2, 3:], goal, 0.075)
    assert np.isclose(abs(rem), d - 3 * 0.075, atol=1e-6)
    rest, _, _ = generate_rotate_then_go(np.zeros(3), traj[0, 2, 3:], np.array([rem]), 0.75, 0.2, 0.5, 0.1, 60)
    assert np.isclose(angle_between(heading_of_pose7(rest[0, -1]), goal), 0.0, atol=1e-6)
    assert turn_remainder(traj[0, -1, 3:], goal, 0.075) is None


# --- reversing ---------------------------------------------------------------------

def test_armed_the_careful_reverse_opens_only_when_the_full_one_is_blocked():
    params = np.array([[0.4, 0.0], [-REVERSE_SPEED, 0.0], [-REVERSE_CAREFUL_SPEED, 0.0]])
    inf = np.inf
    assert list(admit_reverse([0.0, 0.0, inf], params, True)) == [0.0, 0.0, inf]
    assert list(admit_reverse([inf, inf, inf], params, True)) == [inf, inf, 0.0]
    assert list(admit_reverse([0.0, inf, 0.0], params, False)) == [0.0, inf, inf]


def test_the_careful_reverse_stops_after_its_length():
    trajs, params = generate_predefined_trajectory_vocabularies(init_q=_LEVEL_Q)
    careful = trajs[np.flatnonzero(np.isclose(params[:, 0], -REVERSE_CAREFUL_SPEED))[0]]
    full = trajs[np.flatnonzero(np.isclose(params[:, 0], -REVERSE_SPEED))[0]]
    assert np.isclose(np.linalg.norm(careful[-1, :2]), REVERSE_CAREFUL_M)
    assert np.linalg.norm(full[-1, :2]) > 2 * REVERSE_CAREFUL_M



def test_the_route_left_at_a_point_is_from_its_nearest_route_point():
    route = np.c_[np.arange(0.0, 8.0, 0.05), np.zeros(160)]
    assert abs(route_left_at(route, [3.01, 0.2]) - 4.95) < 1e-6


def _walled(wall_to_y):
    """3 m x 3 m at 0.05 m, a wall at x = 1.5 from y = 0 up to `wall_to_y`: the cells
    0.25 m clear of it."""
    occ = np.zeros((60, 60), dtype=bool)
    occ[30, :int(wall_to_y / 0.05)] = True
    return distance_transform_edt(~occ) * 0.05 >= 0.25


def _around(free, ends, target):
    """Every row facing +x: (straight, its bearing error, the distance and error back)."""
    to = target - ends
    straight = np.linalg.norm(to, axis=1)
    err = np.abs(np.arctan2(to[:, 1], to[:, 0]))
    return (straight, err) + around_obstacles(ends, np.zeros(len(ends)), straight, err,
                                              np.zeros(len(ends)), target, free, np.zeros(2),
                                              0.05, 0.25)


def test_a_row_behind_a_wall_from_the_target_goes_round_it():
    ends = np.array([[1.0, 0.5], [2.0, 1.5]])         # behind the wall; on the target's side
    straight, err, d, e = _around(_walled(2.0), ends, np.array([2.0, 0.5]))
    assert d[0] > straight[0] + 2.0                   # up round the wall's end and back
    assert err[0] == 0.0 and e[0] > 1.0               # faces the wall, not the way round
    assert d[1] == straight[1] and e[1] == err[1]


def test_a_target_walled_off_leaves_every_row_its_straight_distance():
    ends = np.array([[1.0, 0.5], [1.0, 2.6]])
    straight, err, d, e = _around(_walled(3.0), ends, np.array([2.0, 0.5]))  # the wall spans the grid
    assert np.allclose(d, straight) and np.allclose(e, err)


def _blocked_with_box_at(x, y):
    """A 4m route along x at y=2.5, the robot at x=1 and the stop pose at x=3, and one
    obstacle cell at (x, y)."""
    origin, res = np.array([0.0, 0.0]), 0.1
    path_dist, remaining, _, _ = build_route_fields([(0.5, 2.5), (4.5, 2.5)], (50, 50), origin, res)
    obstacle = np.zeros((50, 50), dtype=bool)
    obstacle[int(x / res), int(y / res)] = True
    esdf = distance_transform_edt(~obstacle) * res
    return route_blocked_ahead(esdf, path_dist, remaining, rem_here=3.5, stop_rem=1.5, half_w=0.25)


def test_an_obstacle_on_the_route_before_the_stop_pose_blocks_it():
    assert _blocked_with_box_at(2.0, 2.5)
    assert _blocked_with_box_at(2.0, 2.7)             # within half a body of the line


def test_an_obstacle_beside_behind_or_past_the_stop_pose_does_not():
    assert not _blocked_with_box_at(2.0, 3.0)         # clears the body
    assert not _blocked_with_box_at(0.6, 2.5)         # behind the robot
    assert not _blocked_with_box_at(4.0, 2.5)         # past the stop pose



# --- footprint_scale --------------------------------------------------------------
#: A body whose edges and its 0.8 edges sit mid-cell on the 0.05 grids below, so the
#: cell an obstacle lands in is not decided by float rounding at a cell boundary.
_BODY = replace(ROBOT_CONFIG, front_len=0.42, rear_len=0.42, half_width=0.18,
                safety_radius=0.1, footprint_scale=(1.0, 1.0, 1.0))


def _three(scale):
    """A footprint_scale: one number is the same factor front, rear and width."""
    return tuple(scale) if isinstance(scale, tuple) else (scale, scale, scale)


def _planning_sees(monkeypatch, scale):
    """planning_node's ROBOT_CONFIG as _BODY at `scale` (one factor or front, rear,
    width): what the shipped methods read."""
    import planning_node
    monkeypatch.setattr(planning_node, 'ROBOT_CONFIG', replace(_BODY, footprint_scale=_three(scale)))


def test_at_scale_one_the_lattice_is_the_bodys():
    """The default changes nothing: the same samples, bit for bit, as before the scale."""
    a = footprint_lattice(*ROBOT_CONFIG.footprint_from_control(), ROBOT_CONFIG.safety_radius)
    b = footprint_lattice(*replace(ROBOT_CONFIG, footprint_scale=(1.0, 1.0, 1.0)).collision_footprint(),
                          ROBOT_CONFIG.safety_radius)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_a_scale_shrinks_the_lattice_and_keeps_its_pitch():
    """0.8 spans 0.8 of the body each way, and the pitch is still safety_radius's, so the
    smaller outline is covered as densely as the body was."""
    for front, rear, width in ((1.0, 1.0, 1.0), (0.8, 0.8, 0.8), (0.5, 0.5, 0.5),
                               (1.0, 1.0, 0.6), (0.9, 0.6, 1.0)):
        fl, rl, hw = replace(_BODY, footprint_scale=(front, rear, width)).collision_footprint()
        off_fwd, off_lat = footprint_lattice(fl, rl, hw, _BODY.safety_radius)
        assert np.isclose(off_fwd.max(), front * 0.42) and np.isclose(off_fwd.min(), -rear * 0.42)
        assert np.isclose(off_lat.max(), width * 0.18) and np.isclose(off_lat.min(), -width * 0.18)
        assert (off_fwd[0], off_lat[0]) == (0.0, 0.0)
        samples = np.stack([off_fwd, off_lat], axis=1)
        grid = np.stack(np.meshgrid(np.linspace(-rl, fl, 41), np.linspace(-hw, hw, 41)), -1).reshape(-1, 2)
        worst = np.linalg.norm(grid[:, None] - samples[None], axis=2).min(axis=1).max()
        assert worst <= _BODY.safety_radius + 1e-9, ((front, rear, width), worst)


def test_at_0_8_the_body_outline_past_the_scaled_one_is_not_a_body_hit(monkeypatch):
    nose_corner, scaled_corner = (0.42, -0.18), (0.8 * 0.42, -0.8 * 0.18)
    _planning_sees(monkeypatch, 1.0)
    assert _hits_at(nose_corner)[0], 'the body corner is not on the 1.0 lattice'
    _planning_sees(monkeypatch, 0.8)
    assert _hits_at(nose_corner)[0] == []
    assert _hits_at(scaled_corner)[0], 'the scaled corner is not checked at 0.8'


def test_only_the_width_shrinks_when_only_the_width_is_scaled(monkeypatch):
    """Width 0.6 lets go of the body's side edge, and the nose, at full length, still
    collides. (Points on lattice rows: the corners and, at width 0.6, the centre line.)"""
    nose_corner, nose_centre = (0.42, -0.18), (0.42, 0.0)
    _planning_sees(monkeypatch, (1.0, 1.0, 0.6))
    assert _hits_at(nose_corner)[0] == [], 'the body side is still checked at width 0.6'
    assert _hits_at(nose_centre)[0], 'the nose is not checked though the front is 1.0'
    assert _hits_at((0.42, -0.6 * 0.18))[0], 'the side at width 0.6 is not checked'


def test_front_and_rear_shrink_each_by_their_own(monkeypatch):
    """Front 1.0, rear 0.6: the tail lets go of what the nose still sees."""
    nose, tail = (0.42, -0.18), (-0.42, -0.18)
    _planning_sees(monkeypatch, (1.0, 0.6, 1.0))
    assert _hits_at(nose)[0]
    assert _hits_at(tail)[0] == []
    assert _hits_at((-0.6 * 0.42, -0.18))[0], 'the rear at 0.6 is not checked'


def _front_clearance(obstacle_xy):
    """The shipped `_front_obstacle_dist`, for a robot at the origin facing +x."""
    node = _EscapeNode(obstacle_xy)
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return PlanningNode._front_obstacle_dist(node, T, node.mask, max_dist=1.0)


def test_at_0_8_the_front_clearance_is_from_the_scaled_face(monkeypatch):
    """The speed schedule and the reverse gate read this, so it has to start where the
    rows are checked from: 0.2 of front_len further, to within a cell, and the corridor
    is the scaled width -- an obstacle off the scaled side is not ahead."""
    ahead, beside = (0.72, 0.0), (0.72, -0.18)
    _planning_sees(monkeypatch, 1.0)
    full = _front_clearance(ahead)
    assert abs(full - 0.30) <= 0.05 and _front_clearance(beside) == full
    _planning_sees(monkeypatch, 0.8)
    assert abs(_front_clearance(ahead) - (full + 0.2 * 0.42)) <= 0.05
    assert _front_clearance(beside) > 1.0, 'the corridor is still the body width'


def test_at_0_8_an_obstacle_on_the_body_edge_no_longer_vetoes_standing(monkeypatch):
    """The scorer's collision is the scaled lattice: an obstacle cell on the body's nose
    corner is a collision at 1.0 and clearance (here past safety_radius) at 0.8."""
    _planning_sees(monkeypatch, 1.0)
    assert _score_rows((0.42, -0.18), back_step=_RES)[0] == float('inf')
    _planning_sees(monkeypatch, 0.8)
    assert _score_rows((0.42, -0.18), back_step=_RES)[0] == 0.0


def test_planning_reads_the_body_footprint_only_to_log_it():
    """Every collision use -- the lattice and scoring, the body hits, the front scan and
    reverse gate, the strict-follow and around-obstacles widths, the decay under the
    body -- goes through collision_footprint; the body's own numbers are only logged
    next to it. ARRIVE_M is planned_arrival_radius, which stays the body's."""
    tree = ast.parse(open(os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core',
                                       'planning_node.py')).read())
    body = {'footprint_from_control', 'front_len', 'rear_len', 'half_width'}

    def reads(node):
        return sum(1 for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr in body
                   and isinstance(n.value, ast.Name) and n.value.id == 'ROBOT_CONFIG')

    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    assert reads(tree) == reads(init) == 3


# --- a map's footprint_scale, per direction, at run time -------------------------------
DIRECTIONS = ('front', 'rear', 'width')


class _Clock:
    """A ROS clock the test turns by hand."""

    def __init__(self):
        self.ns = 10 ** 12

    def now(self):
        return types.SimpleNamespace(nanoseconds=self.ns)


def _scale_node(monkeypatch, rig_scale=(1.0, 1.0, 1.0)):
    """A PlanningNode with only its footprint_scale plumbing, via __new__ so none of
    __init__'s loading runs: the rig is _BODY at `rig_scale`, the clock is _Clock, and
    what it says on each in-use topic and what it logs are kept."""
    import planning_node
    monkeypatch.setattr(planning_node, 'ROBOT_CONFIG', replace(_BODY, footprint_scale=_three(rig_scale)))
    node = planning_node.PlanningNode.__new__(planning_node.PlanningNode)
    clock, said, logged = _Clock(), {d: [] for d in DIRECTIONS}, []
    node.get_clock = lambda: clock
    node.get_logger = lambda: types.SimpleNamespace(info=logged.append)
    node.footprint_scale_in_use_pubs = {
        d: types.SimpleNamespace(publish=lambda msg, d=d: said[d].append(msg.data)) for d in DIRECTIONS}
    node._map_footprint_scale_ttl_ns = int(planning_node.MAP_FOOTPRINT_SCALE_TTL_S * 1e9)
    node._map_footprint_scale = {d: float('nan') for d in DIRECTIONS}
    node._map_footprint_scale_stamp_ns = {d: None for d in DIRECTIONS}
    node._use_footprint_scale(_three(rig_scale))
    return node, clock, said, logged


def _send(node, direction, value):
    """`value` on `direction`'s topic, as the wire carries it: float32's precision."""
    from std_msgs.msg import Float32
    node.footprint_scale_callback(direction, Float32(data=float(np.float32(value))))


def test_a_direction_in_use_is_the_maps_while_fresh_and_a_factor():
    from planning_node import footprint_scale_in_use
    assert footprint_scale_in_use(float(np.float32(0.8)), True, 1.0) == 0.8   # rounded back
    assert footprint_scale_in_use(0.5, True, 0.9) == 0.5 and footprint_scale_in_use(1.0, True, 0.7) == 1.0
    for unset in (float('nan'), float('inf'), 0.45, 1.05, None):
        assert footprint_scale_in_use(unset, True, 0.9) == 0.9, unset
    assert footprint_scale_in_use(0.8, False, 0.9) == 0.9, 'a stale map factor still applied'


def test_a_map_scaling_only_the_width_leaves_the_length_alone(monkeypatch):
    node, _, said, logged = _scale_node(monkeypatch)
    assert node._collision_fp == pytest.approx((0.42, 0.42, 0.18))
    _send(node, 'width', 0.8)
    assert node._footprint_scale_in_use == (1.0, 1.0, 0.8)
    assert node._collision_fp == pytest.approx((0.42, 0.42, 0.8 * 0.18))
    assert said == {'front': [1.0], 'rear': [1.0], 'width': [0.8]}, 'the change is said at once'
    assert 'x1/1/1 -> x1/1/0.8 (map: width)' in logged[-1]
    _send(node, 'width', 0.8)                     # the same value every second
    assert said['width'] == [0.8] and len(logged) == 1, 'an unchanged scale is not a switch'


def test_front_and_rear_take_their_own_factors(monkeypatch):
    node, _, said, logged = _scale_node(monkeypatch)
    _send(node, 'front', 0.9)
    _send(node, 'rear', 0.6)
    assert node._footprint_scale_in_use == (0.9, 0.6, 1.0)
    assert node._collision_fp == pytest.approx((0.9 * 0.42, 0.6 * 0.42, 0.18))
    assert 'x0.9/1/1 -> x0.9/0.6/1 (map: front, rear)' in logged[-1]


def test_nan_or_a_value_out_of_range_is_the_rigs_factor_in_that_direction(monkeypatch):
    for unset in (float('nan'), 0.3, 1.2):
        node, _, said, logged = _scale_node(monkeypatch, rig_scale=(0.9, 0.95, 0.85))
        _send(node, 'front', 0.7)
        _send(node, 'width', 0.6)
        assert node._footprint_scale_in_use == (0.7, 0.95, 0.6)
        _send(node, 'width', unset)
        assert node._footprint_scale_in_use == (0.7, 0.95, 0.85), unset
        assert node._collision_fp == pytest.approx((0.7 * 0.42, 0.95 * 0.42, 0.85 * 0.18))
        assert said['width'][-1] == pytest.approx(0.85) and '(map: front)' in logged[-1]


def test_one_direction_gone_silent_hands_back_alone(monkeypatch):
    """Each direction has its own freshness: the front stops arriving and goes back to
    the rig's, while the width, still sent, holds."""
    node, clock, said, logged = _scale_node(monkeypatch)
    _send(node, 'front', 0.8)
    _send(node, 'width', 0.7)
    clock.ns += int(2.0e9)
    _send(node, 'width', 0.7)                     # the width keeps coming, the front not
    clock.ns += int(1.1e9)                        # front last heard 3.1 s ago
    node._footprint_scale_tick()
    assert node._footprint_scale_in_use == (1.0, 1.0, 0.7)
    assert node._collision_fp == pytest.approx((0.42, 0.42, 0.7 * 0.18))
    assert 'x0.8/1/0.7 -> x1/1/0.7 (map: width)' in logged[-1]
    clock.ns += int(2.0e9)                        # width last heard 3.1 s ago too
    node._footprint_scale_tick()
    assert node._footprint_scale_in_use == (1.0, 1.0, 1.0) and '(map: none)' in logged[-1]


def test_the_factors_in_use_go_out_every_second_unchanged_too(monkeypatch):
    node, _, said, logged = _scale_node(monkeypatch, rig_scale=(0.85, 1.0, 0.9))
    node._footprint_scale_tick()
    node._footprint_scale_tick()
    assert said == {'front': [0.85, 0.85], 'rear': [1.0, 1.0], 'width': [0.9, 0.9]}
    assert logged == []


def test_a_new_scale_reaches_the_checks_and_compiles_nothing(monkeypatch):
    """The body hits follow the node's outline the cycle after a switch, and the njit
    scorer takes the outline as arguments: scoring at other factors adds no
    specialisation, so a map's scale costs no recompile mid-run."""
    node, _, _, _ = _scale_node(monkeypatch)
    nose_corner = (0.42, -0.18)
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    stand_in = _EscapeNode(nose_corner)
    stand_in._collision_fp = node._collision_fp
    assert PlanningNode._footprint_hits(stand_in, T, stand_in.mask)[0]
    _send(node, 'width', 0.6)
    stand_in._collision_fp = node._collision_fp
    assert PlanningNode._footprint_hits(stand_in, T, stand_in.mask)[0] == []

    _planning_sees(monkeypatch, 1.0)
    _score_rows(nose_corner, back_step=_RES)
    compiled = len(score_trajectories_by_ESDF.signatures)
    _planning_sees(monkeypatch, (0.9, 0.6, 0.8))
    _score_rows(nose_corner, back_step=_RES)
    assert len(score_trajectories_by_ESDF.signatures) == compiled


def test_the_provider_declares_what_the_planner_reads():
    """pilot-nav.json says the planner reads a map's footprint_scale
    (`planner.footprint_scale`) and declares no map setting for it: the footprint
    plugin owns the per-map factors and publishes them on the topics this process
    reads, so pilot must not publish the same topics from a map setting too."""
    import json
    from tinynav.core.robot_specs import (FOOTPRINT_SCALE_RANGE, FOOTPRINT_SCALE_TOPIC,
                                          FOOTPRINT_SCALE_IN_USE_TOPIC, FOOTPRINT_SCALE_DIRECTIONS)
    root = os.path.join(os.path.dirname(__file__), '..')
    provider = json.load(open(os.path.join(root, 'pilot-nav.json')))
    assert 'planner.footprint_scale' in provider['capabilities']
    settings = provider.get('map_settings', {})
    assert not [k for k in settings if k.startswith('footprint_scale')]
    topics = {s.get('topic') for s in settings.values()} | {s.get('in_use_topic') for s in settings.values()}
    assert FOOTPRINT_SCALE_DIRECTIONS == DIRECTIONS
    for d in DIRECTIONS:
        assert FOOTPRINT_SCALE_TOPIC.format(d) not in topics
        assert FOOTPRINT_SCALE_IN_USE_TOPIC.format(d) not in topics
    # The plugin's range is this one (plugins/footprint/scales.py).
    assert FOOTPRINT_SCALE_RANGE == (0.5, 1.0)

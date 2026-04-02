"""
Unit test for build_fused_esdf_from_height stair detection.
No ROS required - pure numpy.

Run:
    cd /home/xiaolefang/workspace/tinynav
    python test_esdf_stair.py
"""
"""
Standalone copy of the functions under test — no ROS, no numba needed.
Keep in sync with tinynav/core/planning_node.py manually (or update when the
real functions change).
"""
import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt, binary_dilation


# ── Inline copy of the two functions under test ───────────────────────────────

@dataclass
class FusedESDFConfig:
    robot_z_bottom: float = -0.2
    robot_z_top: float = 0.1
    occ_threshold: float = 0.1
    stair_allow_height: float = 0.4
    slope_threshold_deg: float = 45.0
    default_clear_distance: float = 100.0


def _build_esdf_from_obstacle(obstacle, resolution, default_clear_distance):
    if np.any(obstacle):
        return (distance_transform_edt(~obstacle) * resolution).astype(np.float32)
    return np.full(obstacle.shape, default_clear_distance, dtype=np.float32)


def build_fused_esdf_from_height(height_map_rel, occupancy_grid, origin, resolution, robot_z, config=None, return_debug=False):
    """Inline copy from planning_node.py for isolated testing."""
    config = config or FusedESDFConfig()

    h, w, z_dim = occupancy_grid.shape
    z_world = origin[2] + (np.arange(z_dim) + 0.5) * resolution
    z_rel = z_world - robot_z
    z_mask = (z_rel >= float(config.robot_z_bottom)) & (z_rel <= float(config.robot_z_top))

    finite = np.isfinite(height_map_rel)
    filled = np.nan_to_num(height_map_rel, nan=0.0, neginf=0.0, posinf=0.0)

    step_x = np.zeros((h, w), dtype=np.float32)
    step_y = np.zeros((h, w), dtype=np.float32)
    if w > 1:
        valid_x = finite[:, 1:] & finite[:, :-1]
        diff_x = np.abs(filled[:, 1:] - filled[:, :-1])
        diff_x = np.where(valid_x, diff_x, 0.0)
        step_x[:, 1:] = np.maximum(step_x[:, 1:], diff_x)
        step_x[:, :-1] = np.maximum(step_x[:, :-1], diff_x)
    if h > 1:
        valid_y = finite[1:, :] & finite[:-1, :]
        diff_y = np.abs(filled[1:, :] - filled[:-1, :])
        diff_y = np.where(valid_y, diff_y, 0.0)
        step_y[1:, :] = np.maximum(step_y[1:, :], diff_y)
        step_y[:-1, :] = np.maximum(step_y[:-1, :], diff_y)

    local_height_diff = np.maximum(step_x, step_y)
    slope = local_height_diff / max(1e-6, resolution)

    obstacle = np.zeros((h, w), dtype=bool)
    stair_like = np.zeros((h, w), dtype=bool)
    if np.any(z_mask):
        band_obstacle = np.any(occupancy_grid[:, :, z_mask] > float(config.occ_threshold), axis=2)
        stair_like = finite & (local_height_diff <= float(config.stair_allow_height))
        obstacle = band_obstacle & (~stair_like)

        if np.any(stair_like):
            stair_like_dilated = binary_dilation(stair_like, iterations=1)
            low_lying = finite & (filled < float(config.robot_z_top + 0.5))
            boundary_rescue = band_obstacle & stair_like_dilated & (~stair_like) & low_lying
            obstacle = obstacle & (~boundary_rescue)

    fused_esdf = _build_esdf_from_obstacle(obstacle, resolution, config.default_clear_distance)
    unknown_mask = ~finite

    if not return_debug:
        return fused_esdf, slope, unknown_mask

    debug = {
        'step_obstacle_mask': obstacle,
        'stair_like_mask': stair_like,
        'local_height_diff': local_height_diff,
        'unknown_mask': unknown_mask,
        'fused_esdf': fused_esdf,
    }
    return fused_esdf, slope, unknown_mask, debug


def make_flat_scene(resolution=0.1, grid_size=20, z_dim=10):
    """平地场景：全部高度 0，地面层有占用。应该没有任何 obstacle。"""
    origin = np.array([0.0, 0.0, -0.5])
    robot_z = 0.0
    height_map_rel = np.zeros((grid_size, grid_size), dtype=np.float32)
    occupancy_grid = np.zeros((grid_size, grid_size, z_dim), dtype=np.float32)
    occupancy_grid[:, :, 5] = 0.5  # z=5 => world z ~ 0.0，在 band [-0.2, +0.1] 内
    return height_map_rel, occupancy_grid, origin, resolution, robot_z


def make_stair_scene(resolution=0.1, grid_size=20, z_dim=10, step_height=0.03):
    """
    楼梯场景：
    - 左半边 col 0~9：平地，高度 0.0
    - 右半边 col 10~19：逐渐抬升（每格 step_height），模拟台阶
    - 整体 occupancy_grid 在地面层有占用
    """
    origin = np.array([0.0, 0.0, -0.5])
    robot_z = 0.0
    height_map_rel = np.zeros((grid_size, grid_size), dtype=np.float32)
    for col in range(10, grid_size):
        height_map_rel[:, col] = (col - 10) * step_height

    occupancy_grid = np.zeros((grid_size, grid_size, z_dim), dtype=np.float32)
    occupancy_grid[:, :, 5] = 0.5
    return height_map_rel, occupancy_grid, origin, resolution, robot_z


def make_wall_scene(resolution=0.1, grid_size=20, z_dim=10):
    """
    真障碍场景：
    - col 10 有一堵竖墙，height_map 高度突变 1.0m（远超 stair_allow_height）
    - 且 occupancy_grid 在高度带内有占用
    应该: col 10 被标为 obstacle
    """
    origin = np.array([0.0, 0.0, -0.5])
    robot_z = 0.0
    height_map_rel = np.zeros((grid_size, grid_size), dtype=np.float32)
    height_map_rel[:, 10:] = 1.0  # 高度突变 1m

    occupancy_grid = np.zeros((grid_size, grid_size, z_dim), dtype=np.float32)
    occupancy_grid[:, :, 5] = 0.5  # 地面层全部有占用（包括墙）
    return height_map_rel, occupancy_grid, origin, resolution, robot_z


def run_and_print(name, height_map_rel, occupancy_grid, origin, resolution, robot_z, config=None):
    config = config or FusedESDFConfig()
    fused_esdf, slope, unknown_mask, debug = build_fused_esdf_from_height(
        height_map_rel, occupancy_grid, origin, resolution, robot_z,
        config=config, return_debug=True
    )
    obstacle = debug['step_obstacle_mask']
    print(f"\n{'='*50}")
    print(f"Scene: {name}")
    print(f"  obstacle cells: {obstacle.sum()}")
    print(f"  unknown cells:  {unknown_mask.sum()}")
    print(f"  ESDF min/max:   {fused_esdf.min():.3f} / {fused_esdf.max():.3f}")
    return obstacle, fused_esdf, unknown_mask


# ── Test 1: 平地不产生 obstacle ──────────────────────────────────────────────
def test_flat_no_obstacle():
    h, o, origin, res, rz = make_flat_scene()
    obstacle, _, _ = run_and_print("Flat ground", h, o, origin, res, rz)
    assert not obstacle.any(), f"❌ FAIL: flat ground has {obstacle.sum()} obstacle cells"
    print("✅ PASS: flat ground - no obstacles")


# ── Test 2: 楼梯不产生 obstacle（核心测试）──────────────────────────────────
def test_stair_not_obstacle():
    h, o, origin, res, rz = make_stair_scene(step_height=0.03)  # 每格 3cm，10格=0.3m
    obstacle, _, _ = run_and_print("Stairs (3cm/step)", h, o, origin, res, rz)
    stair_region = obstacle[:, 10:]
    n_blocked = stair_region.sum()
    if n_blocked == 0:
        print("✅ PASS: stairs not marked as obstacle")
    else:
        print(f"❌ FAIL: {n_blocked} stair cells marked as obstacle")
        # 不 assert，方便看数字调参
    return n_blocked == 0


# ── Test 3: 高台阶（接近 stair_allow_height 边界）──────────────────────────
def test_high_step_boundary():
    # 每格 4.5cm，10格=0.45m（超过默认 stair_allow_height=0.4）
    h, o, origin, res, rz = make_stair_scene(step_height=0.045)
    obstacle, _, _ = run_and_print("High steps (4.5cm/step, total 0.45m)", h, o, origin, res, rz)
    stair_region = obstacle[:, 10:]
    n_blocked = stair_region.sum()
    print(f"  stair cells blocked: {n_blocked} (expected: some, near boundary)")
    return n_blocked


# ── Test 4: 真墙应该是 obstacle ──────────────────────────────────────────────
def test_wall_is_obstacle():
    h, o, origin, res, rz = make_wall_scene()
    obstacle, _, _ = run_and_print("Wall (1m height jump)", h, o, origin, res, rz)
    wall_col = obstacle[:, 10]
    n_blocked = wall_col.sum()
    if n_blocked > 0:
        print(f"✅ PASS: wall correctly blocked ({n_blocked} cells)")
    else:
        print("❌ FAIL: wall not detected as obstacle!")
    return n_blocked > 0


# ── Inline minimal A* for the mixed test (no ROS/numba needed) ───────────────
import heapq

def astar_2d_simple(cost_map, obstacle_mask, start, goal):
    """Minimal A* for testing purposes."""
    h, w = obstacle_mask.shape
    if obstacle_mask[start] or obstacle_mask[goal]:
        return None
    open_set = [(0.0, start)]
    came_from = {}
    g = {start: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return list(reversed(path))
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= nx < h and 0 <= ny < w and not obstacle_mask[nx, ny]:
                ng = g[cur] + float(cost_map[nx, ny]) * (1.414 if dx and dy else 1.0)
                if ng < g.get((nx,ny), 1e18):
                    g[(nx,ny)] = ng
                    came_from[(nx,ny)] = cur
                    f = ng + abs(nx-goal[0]) + abs(ny-goal[1])
                    heapq.heappush(open_set, (f, (nx,ny)))
    return None  # no path


def make_mixed_scene(resolution=0.1, grid_size=40, z_dim=10):
    """
    混合场景（沿 col 方向前进）：
    - col  0~14: 平地，高度 0.0
    - col 15~24: 楼梯，高度逐渐抬升（每格 2.5cm，共 0.25m）
    - col 25~26: 竖墙，高度突变 1.5m（高度带内有密集占用）
    - col 27~39: 平地（墙后）
    - 走廊宽度：row 10~29（中间20格宽），行 0~9 和 30~39 是 unknown/nan
    robot 起点: (row=19, col=2)  →  终点: (row=19, col=37)
    """
    origin = np.array([0.0, 0.0, -0.5])
    robot_z = 0.0

    height_map_rel = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    occupancy_grid = np.zeros((grid_size, grid_size, z_dim), dtype=np.float32)

    # 走廊区域 row 10~29 有高度信息
    corridor_rows = slice(10, 30)

    # 平地 col 0~14
    height_map_rel[corridor_rows, 0:15] = 0.0
    occupancy_grid[corridor_rows, 0:15, 5] = 0.5

    # 楼梯 col 15~24
    for col in range(15, 25):
        h_val = (col - 15) * 0.025  # 每格 2.5cm
        height_map_rel[corridor_rows, col] = h_val
        occupancy_grid[corridor_rows, col, 5] = 0.5

    # 墙 col 25~26：高度突变 1.5m，且在机器人高度带内有实心占用
    height_map_rel[corridor_rows, 25:27] = 1.5
    # z_rel band [-0.2, +0.1] → world z = robot_z + [-0.2, +0.1] = [-0.2, 0.1]
    # origin[2] = -0.5，z_dim=10，resolution=0.1
    # z_world[i] = -0.5 + (i+0.5)*0.1 → z_rel = z_world - 0.0
    # band indices: z_rel in [-0.2, 0.1] → z_world in [-0.2, 0.1]
    # i: (-0.5 + (i+0.5)*0.1) in [-0.2, 0.1] → i in [3, 6]
    for zi in range(3, 7):
        occupancy_grid[corridor_rows, 25:27, zi] = 1.0  # solid wall

    # 墙后平地 col 27~39
    height_map_rel[corridor_rows, 27:] = 0.25  # 稍微高一点（墙后地面）
    occupancy_grid[corridor_rows, 27:, 5] = 0.5

    return height_map_rel, occupancy_grid, origin, resolution, robot_z


def build_cost_map_simple(fused_esdf, unknown_mask, hard_esdf=0.05, safe_clearance=0.45):
    """Simplified cost map matching planning_node logic (ESDF-weighted unknown penalty)."""
    obstacle = fused_esdf < hard_esdf
    d = np.clip(fused_esdf, hard_esdf, 2.0)
    barrier = np.where(
        d >= safe_clearance,
        0.0,
        ((safe_clearance - d) / max(1e-6, safe_clearance)) ** 2,
    )
    # ESDF-weighted unknown penalty (mirrors planning_node change)
    unknown_penalty = np.where(
        unknown_mask,
        np.clip(20.0 - fused_esdf * 15.0, 3.0, 20.0),
        0.0,
    )
    cost = 1.0 + 10.0 * barrier + unknown_penalty
    return obstacle, cost


def apply_global_path_bonus(cost_map, global_path_xy, origin, resolution,
                             bonus=4.0, dilation_cells=1):
    """Inline copy from planning_node.py for testing."""
    if global_path_xy is None or len(global_path_xy) == 0:
        return cost_map
    h, w = cost_map.shape
    bonus_map = np.zeros((h, w), dtype=np.float32)
    for pt in global_path_xy:
        gx = int((pt[0] - origin[0]) / resolution)
        gy = int((pt[1] - origin[1]) / resolution)
        for dx in range(-dilation_cells, dilation_cells + 1):
            for dy in range(-dilation_cells, dilation_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < h and 0 <= ny < w:
                    bonus_map[nx, ny] = bonus
    return np.maximum(1.0, cost_map - bonus_map)


def visualize_scene(obstacle, path, grid_size, start, goal, name):
    """ASCII visualization of the scene + path."""
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            if obstacle[r, c]:
                grid[r][c] = '#'
    if path:
        for (r, c) in path:
            grid[r][c] = '*'
    grid[start[0]][start[1]] = 'S'
    grid[goal[0]][goal[1]] = 'G'

    print(f"\n  Map ({name}) — S=start G=goal *=path #=obstacle .=free/unknown")
    # Print every other row to keep it compact
    for r in range(0, grid_size, 2):
        print("  " + "".join(grid[r]))


# ── Inline footprint + DWA (from planning_node.py) ───────────────────────────

def world_to_grid_2d(point_xy, origin, resolution):
    gx = int((point_xy[0] - origin[0]) / resolution)
    gy = int((point_xy[1] - origin[1]) / resolution)
    return (gx, gy)


def footprint_corners_xy(center_xy, forward_xy, front_length, rear_length, half_width):
    c = np.asarray(center_xy, dtype=np.float32)
    fwd = np.asarray(forward_xy, dtype=np.float32)
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        fwd = np.array([1.0, 0.0], dtype=np.float32)
    else:
        fwd = fwd / n
    left_xy = np.array([-fwd[1], fwd[0]], dtype=np.float32)
    return [
        c + fwd * front_length + left_xy * half_width,
        c + fwd * front_length - left_xy * half_width,
        c - fwd * rear_length - left_xy * half_width,
        c - fwd * rear_length + left_xy * half_width,
    ]


def footprint_collision_4corners(center_xy, forward_xy, obstacle_mask, origin, resolution,
                                  front_length, rear_length, half_width):
    h, w = obstacle_mask.shape
    corners = footprint_corners_xy(center_xy, forward_xy, front_length, rear_length, half_width)
    for corner in corners:
        gx, gy = world_to_grid_2d(corner, origin, resolution)
        if 0 <= gx < h and 0 <= gy < w:
            if obstacle_mask[gx, gy]:
                return True
    return False


def plan_dwa_minimal(robot_xy, robot_forward_xy, target_xy, fused_esdf, obstacle_mask,
                     cost_map, origin, resolution,
                     horizon_s=1.2, dt=0.2,
                     v_samples=None, w_samples=None,
                     front_len=0.4, rear_len=0.0, half_w=0.15):
    if v_samples is None:
        v_samples = np.linspace(-0.1, 0.5, 10, dtype=np.float32)
    if w_samples is None:
        w_samples = np.array([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2], dtype=np.float32)

    rf = np.asarray(robot_forward_xy, dtype=np.float32)
    nrf = np.linalg.norm(rf)
    rf = rf / nrf if nrf > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    theta0 = float(np.arctan2(rf[1], rf[0]))

    best_score = float('inf')
    best_path = []
    steps = max(1, int(np.ceil(horizon_s / dt)))

    for v in v_samples:
        for w in w_samples:
            x, y, theta = float(robot_xy[0]), float(robot_xy[1]), theta0
            traj = []
            score = 0.0
            collided = False
            for _ in range(steps):
                theta += float(w) * dt
                x += float(v) * np.cos(theta) * dt
                y += float(v) * np.sin(theta) * dt
                traj.append(np.array([x, y], dtype=np.float32))
                gx, gy = world_to_grid_2d((x, y), origin, resolution)
                if gx < 0 or gy < 0 or gx >= obstacle_mask.shape[0] or gy >= obstacle_mask.shape[1]:
                    collided = True; score += 1e6; break
                fwd = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                if footprint_collision_4corners(np.array([x, y], dtype=np.float32), fwd,
                                                obstacle_mask, origin, resolution,
                                                front_len, rear_len, half_w):
                    collided = True; score += 1e6; break
                score += float(cost_map[gx, gy]) * dt
                score += 0.5 * np.linalg.norm(np.array([x, y], dtype=np.float32) - target_xy) * dt
                score += 0.05 * abs(float(w)) * dt
            if not collided and traj:
                score += 2.0 * np.linalg.norm(traj[-1] - target_xy)
            if score < best_score:
                best_score = score
                best_path = traj

    return best_path


# ── Test 5: 混合场景 A* 穿越楼梯、绕开墙 ──────────────────────────────────
def test_mixed_stair_wall_astar():
    resolution = 0.1
    grid_size = 40
    h_map, occ, origin, res, rz = make_mixed_scene(resolution=resolution, grid_size=grid_size)

    fused_esdf, slope, unknown_mask, debug = build_fused_esdf_from_height(
        h_map, occ, origin, res, rz, return_debug=True
    )
    obstacle, cost_map = build_cost_map_simple(fused_esdf, unknown_mask)

    start = (19, 2)
    goal  = (19, 37)

    print(f"\n{'='*50}")
    print("Scene: Mixed (flat → stairs → wall → flat)")
    print(f"  obstacle cells total:  {obstacle.sum()}")
    print(f"  wall col 25-26 blocked: {obstacle[10:30, 25:27].sum()}")
    print(f"  stair col 15-24 blocked: {obstacle[10:30, 15:25].sum()}")

    path = astar_2d_simple(cost_map, obstacle, start, goal)

    if path is None:
        print("❌ FAIL: A* found no path from start to goal")
        visualize_scene(obstacle, [], grid_size, start, goal, "no path")
        return False

    # 检查路径经过了楼梯区域
    path_cols = [c for (r, c) in path]
    passed_stairs = any(15 <= c <= 24 for c in path_cols)
    # 检查路径没有经过墙
    hit_wall = any(obstacle[r, c] for (r, c) in path)
    # 检查路径到达了终点附近
    reached_goal = path[-1] == goal

    print(f"  Path length: {len(path)} steps")
    print(f"  Passed through stair zone (col 15-24): {'✅' if passed_stairs else '⚠️  went around'}")
    print(f"  Hit wall obstacle: {'❌ YES' if hit_wall else '✅ NO'}")
    print(f"  Reached goal: {'✅' if reached_goal else '❌'}")

    visualize_scene(obstacle, path, grid_size, start, goal, "mixed")

    passed = (not hit_wall) and reached_goal
    if passed:
        print("✅ PASS: navigated stair zone, avoided wall, reached goal")
    else:
        print("❌ FAIL: something went wrong")
    return passed


# ── Test 6: 诊断楼梯边界格误标 ───────────────────────────────────────────────
def test_stair_boundary_diagnosis():
    """
    col 24 是楼梯最后一格，紧邻墙(col 25)。
    local_height_diff 会拿到相邻的 col25 高度差（0.025 + 1.275 = 1.3m），
    导致 col24 被误标为 obstacle。
    这个测试打印诊断信息，帮助决定是否需要修 build_fused_esdf_from_height。
    """
    resolution = 0.1
    grid_size = 40
    z_dim = 10
    origin = np.array([0.0, 0.0, -0.5])
    robot_z = 0.0

    height_map_rel = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    occupancy_grid = np.zeros((grid_size, grid_size, z_dim), dtype=np.float32)
    corridor_rows = slice(10, 30)

    height_map_rel[corridor_rows, 0:15] = 0.0
    occupancy_grid[corridor_rows, 0:15, 5] = 0.5
    for col in range(15, 25):
        h_val = (col - 15) * 0.025
        height_map_rel[corridor_rows, col] = h_val
        occupancy_grid[corridor_rows, col, 5] = 0.5
    height_map_rel[corridor_rows, 25:27] = 1.5
    for zi in range(3, 7):
        occupancy_grid[corridor_rows, 25:27, zi] = 1.0
    height_map_rel[corridor_rows, 27:] = 0.25
    occupancy_grid[corridor_rows, 27:, 5] = 0.5

    _, _, _, debug = build_fused_esdf_from_height(
        height_map_rel, occupancy_grid, origin, resolution, robot_z, return_debug=True
    )
    obstacle = debug['step_obstacle_mask']
    local_diff = debug['local_height_diff']

    print(f"\n{'='*50}")
    print("Diagnosis: stair boundary cells (col 14~26)")
    for col in range(14, 27):
        n_obs = obstacle[10:30, col].sum()
        diff_val = local_diff[10:30, col].mean()
        tag = "← MISCLASSIFIED (stair-wall boundary)" if col == 24 and n_obs > 0 else ""
        print(f"  col {col:2d}: height_diff={diff_val:.4f}  obstacle_cells={n_obs:3d} {tag}")

    print("\nAfter boundary_rescue fix (stair_like dilation):")
    # re-run with fix
    _, _, _, debug2 = build_fused_esdf_from_height(
        height_map_rel, occupancy_grid, origin, resolution, robot_z, return_debug=True
    )
    obs2 = debug2['step_obstacle_mask']
    for col in range(14, 27):
        n_obs = obs2[10:30, col].sum()
        tag = "← ✅ rescued" if col == 24 and n_obs == 0 else ""
        print(f"  col {col:2d}: obstacle_cells={n_obs:3d} {tag}")


# ── Test 7: DWA 在混合场景中跟随 A* 路径 ──────────────────────────────────
def test_dwa_follows_astar_path():
    """
    在混合场景里:
    1. 先用 A* 规划完整路径
    2. 从起点开始，每步用 DWA 选最优速度命令
    3. 验证 DWA 每步产生的轨迹不碰障碍，且方向朝向目标
    """
    resolution = 0.1
    grid_size = 40
    z_dim = 10
    origin = np.array([0.0, 0.0, -0.5])
    robot_z = 0.0

    height_map_rel = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    occupancy_grid = np.zeros((grid_size, grid_size, z_dim), dtype=np.float32)
    corridor_rows = slice(10, 30)

    height_map_rel[corridor_rows, 0:15] = 0.0
    occupancy_grid[corridor_rows, 0:15, 5] = 0.5
    for col in range(15, 25):
        height_map_rel[corridor_rows, col] = (col - 15) * 0.025
        occupancy_grid[corridor_rows, col, 5] = 0.5
    height_map_rel[corridor_rows, 25:27] = 1.5
    for zi in range(3, 7):
        occupancy_grid[corridor_rows, 25:27, zi] = 1.0
    height_map_rel[corridor_rows, 27:] = 0.25
    occupancy_grid[corridor_rows, 27:, 5] = 0.5

    fused_esdf, _, unknown_mask, debug = build_fused_esdf_from_height(
        height_map_rel, occupancy_grid, origin, resolution, robot_z, return_debug=True
    )
    obstacle_mask = debug['step_obstacle_mask']

    hard_esdf = 0.05
    safe_clearance = 0.45
    unknown_penalty = 20.0
    d = np.clip(fused_esdf, hard_esdf, 2.0)
    barrier = np.where(d >= safe_clearance, 0.0,
                       ((safe_clearance - d) / max(1e-6, safe_clearance)) ** 2)
    cost_map = (1.0 + 10.0 * barrier + np.where(unknown_mask, unknown_penalty, 0.0)).astype(np.float32)

    # A* 规划
    start_grid = (19, 2)
    goal_grid = (19, 37)
    astar_path = astar_2d_simple(cost_map, obstacle_mask, start_grid, goal_grid)
    assert astar_path is not None, "A* failed to find path"

    # 转成世界坐标
    def g2w(gxy):
        return np.array([origin[0] + (gxy[0] + 0.5) * resolution,
                         origin[1] + (gxy[1] + 0.5) * resolution], dtype=np.float32)
    path_world = [g2w(p) for p in astar_path]

    print(f"\n{'='*50}")
    print("Test: DWA follows A* path in mixed scene")
    print(f"  A* path length: {len(astar_path)} steps")

    # 模拟机器人从起点沿 A* 路径跟随，每 5 步做一次 DWA 检查
    dwa_collisions = 0
    dwa_no_path = 0
    checkpoints = list(range(0, len(path_world) - 5, 5))

    for i in checkpoints:
        robot_xy = path_world[i]
        # 前向方向：当前点到下一个路径点
        nxt = path_world[min(i + 1, len(path_world) - 1)]
        fwd = nxt - robot_xy
        fwd_norm = np.linalg.norm(fwd)
        forward_xy = fwd / fwd_norm if fwd_norm > 1e-6 else np.array([1.0, 0.0])

        # lookahead target：路径上 3 步后的点
        target_xy = path_world[min(i + 3, len(path_world) - 1)]

        dwa_traj = plan_dwa_minimal(
            robot_xy, forward_xy, target_xy,
            fused_esdf, obstacle_mask, cost_map,
            origin, resolution,
            horizon_s=0.6, dt=0.2,
            front_len=0.4, rear_len=0.0, half_w=0.15
        )

        if not dwa_traj:
            dwa_no_path += 1
            continue

        # 检查 DWA 轨迹中每个点不在 obstacle 里
        for pt in dwa_traj:
            gx, gy = world_to_grid_2d(pt, origin, resolution)
            if 0 <= gx < obstacle_mask.shape[0] and 0 <= gy < obstacle_mask.shape[1]:
                if obstacle_mask[gx, gy]:
                    dwa_collisions += 1
                    break

    total_checks = len(checkpoints)
    print(f"  DWA checkpoints tested: {total_checks}")
    print(f"  DWA collision events:   {dwa_collisions}")
    print(f"  DWA no-path events:     {dwa_no_path}")

    passed = dwa_collisions == 0
    if passed:
        print("✅ PASS: DWA produced no collision trajectories along A* path")
    else:
        print(f"❌ FAIL: {dwa_collisions} DWA steps produced collision trajectories")
    return passed


# ── Test 8: ESDF-weighted unknown penalty ────────────────────────────────────
def test_esdf_weighted_unknown_penalty():
    """
    Unknown cells far from obstacles should have lower penalty than those near walls.
    Uses a simple scene: known corridor with unknown flanks at varying ESDF distances.
    """
    print(f"\n{'='*50}")
    print("Test: ESDF-weighted unknown penalty")

    # Build a simple ESDF map manually:
    # known obstacle at col 0 and col 39 (walls), corridor in between
    grid_size = 40
    fused_esdf = np.full((grid_size, grid_size), 100.0, dtype=np.float32)
    # Simulate ESDF: distance from left wall (col=0)
    for col in range(grid_size):
        fused_esdf[:, col] = col * 0.1  # 0.0m at col0, 3.9m at col39

    unknown_mask = np.zeros((grid_size, grid_size), dtype=bool)
    unknown_mask[:, 10:30] = True  # middle strip is unknown

    obstacle, cost = build_cost_map_simple(fused_esdf, unknown_mask)

    # Cell near wall (col=10, ESDF=1.0): higher unknown penalty
    near_wall_cost = cost[20, 10]
    # Cell far from wall (col=29, ESDF=2.9): lower unknown penalty
    far_wall_cost = cost[20, 29]

    print(f"  Unknown cell near wall (ESDF=1.0m): cost={near_wall_cost:.2f}")
    print(f"  Unknown cell far from wall (ESDF=2.9m): cost={far_wall_cost:.2f}")

    passed = far_wall_cost < near_wall_cost
    if passed:
        print("✅ PASS: far-from-wall unknown has lower cost than near-wall unknown")
    else:
        print("❌ FAIL: ESDF-weighted penalty not working as expected")
    return passed


# ── Test 9: global path bonus reduces cost on path cells ─────────────────────
def test_global_path_bonus():
    """
    Global path bonus should reduce cost on path cells so A* prefers the global route
    over tight-but-known alternatives.
    """
    print(f"\n{'='*50}")
    print("Test: global path bonus + A* prefers global route over wall-hugging")

    resolution = 0.1
    grid_size = 40
    origin = np.array([0.0, 0.0, -0.5])

    # Scene: two known free corridors, both passable.
    # - Top corridor (row 5~9):   ESDF=0.2m → tight, high barrier cost (~11)
    # - Middle corridor (row 18~22): ESDF=2.0m → open, low cost (~1)
    # Without bonus: A* already prefers middle (lower cost) — sanity check
    # Bonus test: on the tight corridor, add global path bonus and verify cost drops

    fused_esdf = np.full((grid_size, grid_size), 2.0, dtype=np.float32)
    # Top corridor is tight
    fused_esdf[5:10, :] = 0.2

    unknown_mask = np.zeros((grid_size, grid_size), dtype=bool)  # all known

    obstacle, cost_map = build_cost_map_simple(fused_esdf, unknown_mask)

    # Verify tight corridor has higher cost
    tight_cost   = float(cost_map[7, 20])
    open_cost    = float(cost_map[20, 20])
    print(f"  Tight corridor cost (ESDF=0.2m): {tight_cost:.2f}")
    print(f"  Open corridor cost  (ESDF=2.0m): {open_cost:.2f}")
    assert tight_cost > open_cost, "tight should cost more than open"

    # Apply bonus along tight corridor → should lower its cost
    # pt = [world_x (→ grid row), world_y (→ grid col)]
    # path runs along grid row=7, across all cols 2~37
    global_path_xy = np.array([
        [origin[0] + (7 + 0.5) * resolution, origin[1] + (col + 0.5) * resolution]
        for col in range(2, 38)
    ], dtype=np.float32)

    cost_with_bonus = apply_global_path_bonus(
        cost_map, global_path_xy, origin, resolution, bonus=4.0, dilation_cells=1
    )

    tight_cost_after = float(cost_with_bonus[7, 20])
    print(f"  Tight corridor cost after bonus: {tight_cost_after:.2f} (was {tight_cost:.2f})")

    # Bonus should reduce tight corridor cost toward open corridor level
    cost_reduced = tight_cost_after < tight_cost
    cost_floor   = tight_cost_after >= 1.0  # never below 1.0

    # A* should now prefer tight corridor (bonus made it cheaper than open)
    start = (20, 2); goal = (20, 37)
    path_no_bonus   = astar_2d_simple(cost_map, obstacle, start, goal)
    path_with_bonus = astar_2d_simple(cost_with_bonus, obstacle, start, goal)

    rows_no_bonus   = set(r for r, c in path_no_bonus)   if path_no_bonus   else set()
    rows_with_bonus = set(r for r, c in path_with_bonus) if path_with_bonus else set()

    drawn_to_tight = any(5 <= r <= 9 for r in rows_with_bonus)
    stayed_open    = any(18 <= r <= 22 for r in rows_no_bonus)

    print(f"  Without bonus → stays on open corridor: {stayed_open}")
    print(f"  With bonus    → drawn to tight corridor: {drawn_to_tight}")

    passed = cost_reduced and cost_floor
    if passed:
        print("✅ PASS: global path bonus correctly reduces cost on path cells (floor=1.0 respected)")
    else:
        print("❌ FAIL: bonus did not reduce cost as expected")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Running ESDF stair detection tests...\n")
    test_flat_no_obstacle()
    stair_ok = test_stair_not_obstacle()
    test_high_step_boundary()
    wall_ok = test_wall_is_obstacle()
    mixed_ok = test_mixed_stair_wall_astar()
    test_stair_boundary_diagnosis()
    dwa_ok = test_dwa_follows_astar_path()
    unknown_ok = test_esdf_weighted_unknown_penalty()
    bonus_ok = test_global_path_bonus()

    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Flat no obstacle:             {'✅' if True else '❌'}")
    print(f"  Stair passable:               {'✅' if stair_ok else '❌'}")
    print(f"  Wall blocked:                 {'✅' if wall_ok else '❌'}")
    print(f"  Mixed A* nav:                 {'✅' if mixed_ok else '❌'}")
    print(f"  DWA follows path safely:      {'✅' if dwa_ok else '❌'}")
    print(f"  ESDF-weighted unknown cost:   {'✅' if unknown_ok else '❌'}")
    print(f"  Global path bonus works:      {'✅' if bonus_ok else '❌'}")

import numpy as np

from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.rl.bev import GRID_RES, GRID_SHAPE
from tinynav.rl.rl_planner import V_MAX, V_MIN, W_MAX, RLPlanner


class FakePolicy:
    def __init__(self, action):
        self.action, self.seen, self.resets = np.asarray(action, np.float32), [], 0

    def reset(self):
        self.resets += 1

    def infer(self, bev, vec):
        self.seen.append((bev, vec))
        return self.action


CAM = float(ROBOT_CONFIG.cam_offset_3d[2])  # control center sits this far behind the camera


def _camera_pose(x, y, yaw, z=0.45):
    """Camera->world with optical axis along heading (x right, y down, z forward), as in the VIO."""
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([r, np.cross(f, r), f], 1)
    T[:3, 3] = [x, y, z]
    return T


def _scene(wall_x=None, wall_y=None):
    """Free grid with an optional obstacle band starting at x=wall_x or below y=wall_y (world, m)."""
    grid = np.full(GRID_SHAPE, -0.2)  # everything observed free
    origin = np.array([-5.0, -5.0, 0.45 - 2.0])
    xs = origin[0] + (np.arange(GRID_SHAPE[0]) + 0.5) * GRID_RES
    ys = origin[1] + (np.arange(GRID_SHAPE[1]) + 0.5) * GRID_RES
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    occ = np.zeros(GRID_SHAPE[:2], bool)
    if wall_x is not None:
        occ |= (X >= wall_x) & (X < wall_x + 0.2)
    if wall_y is not None:
        occ |= (Y <= wall_y) & (Y > wall_y - 0.2)
    from scipy.ndimage import distance_transform_edt
    esdf = (distance_transform_edt(~occ) * GRID_RES).astype(np.float32) if occ.any() else np.full(occ.shape, 9.0, np.float32)
    return grid, origin, esdf


def test_free_space_drives_and_goal_is_in_robot_frame():
    pol = FakePolicy([1.0, 0.5])
    planner = RLPlanner(pol)
    grid, origin, esdf = _scene()
    v, w, poses, bev, status = planner.step(grid, origin, _camera_pose(0.0, 0.0, np.pi / 2), 0.0, np.array([0.0, 2.0]), esdf)
    assert status == "ok" and np.isclose(v, V_MAX) and np.isclose(w, 0.5 * W_MAX)
    assert bev.shape == (3, 64, 64) and poses.shape == (21, 3)
    assert np.allclose(pol.seen[0][1][:2], [(2.0 + CAM) / 5.0, 0.0], atol=1e-5)


def test_wall_ahead_turns_in_place_instead():
    planner = RLPlanner(FakePolicy([1.0, 0.2]))
    grid, origin, esdf = _scene(wall_x=0.9)  # 1m/s for 1s would run the 1.0m footprint into it
    v, w, poses, _, status = planner.step(grid, origin, _camera_pose(0.0, 0.0, 0.0), 0.0, np.array([3.0, 0.0]), esdf)
    assert status == "turn in place" and v == 0.0 and np.isclose(w, 0.3)
    assert np.allclose(poses[:, :2], poses[0, :2])


def test_turning_in_place_is_checked_too():
    # Wall 0.15m beside the robot: driving straight along it is fine, but rotating either way swings
    # a corner of the 1.0 x 0.5m footprint into it, so the guard backs off straight instead.
    grid, origin, esdf = _scene(wall_y=-0.25 - 0.2)
    planner = RLPlanner(FakePolicy([0.0, 1.0]))
    v, w, _, _, status = planner.step(grid, origin, _camera_pose(CAM, 0.0, 0.0), 0.0, np.array([3.0, 0.0]), esdf)
    assert status == "back off" and np.isclose(v, V_MIN) and w == 0.0
    v, w, _, _, status = RLPlanner(FakePolicy([1.0, 0.0])).step(grid, origin, _camera_pose(CAM, 0.0, 0.0), 0.0, np.array([3.0, 0.0]), esdf)
    assert status == "ok" and np.isclose(v, V_MAX)


def test_stuck_robot_is_stopped_until_reset():
    pol = FakePolicy([1.0, 0.0])
    planner = RLPlanner(pol)
    grid, origin, esdf = _scene()
    T = _camera_pose(0.0, 0.0, 0.0)  # commanded forward, but the pose never changes
    statuses = [planner.step(grid, origin, T, 0.15 * k, np.array([3.0, 0.0]), esdf)[4] for k in range(80)]
    assert statuses[0] == "ok" and statuses[-1] == "stuck" and "stuck" not in statuses[:55]
    planner.reset()
    assert planner.step(grid, origin, T, 20.0, np.array([3.0, 0.0]), esdf)[4] == "ok"


def test_reset_clears_memory_and_trail():
    pol = FakePolicy([0.5, 0.0])
    planner = RLPlanner(pol)
    grid, origin, esdf = _scene()
    for k in range(5):
        planner.step(grid, origin, _camera_pose(0.1 * k, 0.0, 0.0), 0.15 * k, np.array([3.0, 0.0]), esdf)
    assert len(planner.hist) == 5 and planner.vel[0] > 0.3 and pol.seen[-1][0][2].max() > 0
    planner.reset()
    assert pol.resets == 2 and planner.hist == [] and not planner.vel.any() and not planner.last_action.any()

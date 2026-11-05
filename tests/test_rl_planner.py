import numpy as np

from tinynav.rl.bev import GRID_RES, GRID_SHAPE
from tinynav.rl.rl_planner import V_MAX, W_MAX, RLPlanner


class FakePolicy:
    def __init__(self, action):
        self.action, self.seen, self.resets = np.asarray(action, np.float32), [], 0

    def reset(self):
        self.resets += 1

    def infer(self, bev, vec):
        self.seen.append((bev, vec))
        return self.action


def _camera_pose(x, y, yaw, z=0.45):
    """Camera->world with optical axis along heading (x right, y down, z forward), as in the VIO."""
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([r, np.cross(f, r), f], 1)
    T[:3, 3] = [x, y, z]
    return T


def _scene(wall_x=None):
    grid = np.full(GRID_SHAPE, -0.2)  # everything observed free
    origin = np.array([-5.0, -5.0, 0.45 - 2.0])
    esdf = np.full(GRID_SHAPE[:2], 5.0, np.float32)
    if wall_x is not None:
        ix = int((wall_x - origin[0]) / GRID_RES)
        esdf[ix:ix + 2, :] = 0.0
    return grid, origin, esdf


def test_free_space_drives_and_goal_is_in_robot_frame():
    pol = FakePolicy([1.0, 0.5])
    planner = RLPlanner(pol)
    grid, origin, esdf = _scene()
    T = _camera_pose(0.0, 0.0, np.pi / 2)  # facing +y
    v, w, poses, bev, guard = planner.step(grid, origin, T, 0.0, np.array([0.0, 2.0]), esdf)
    assert not guard and np.isclose(v, V_MAX) and np.isclose(w, 0.5 * W_MAX)
    assert bev.shape == (3, 64, 64) and poses.shape == (21, 3)
    vec = pol.seen[0][1]
    assert np.allclose(vec[:2], [(2.0 + 0.2) / 5.0, 0.0], atol=1e-5)  # control center sits 0.2m behind the camera


def test_guard_stops_forward_motion_but_keeps_turning():
    planner = RLPlanner(FakePolicy([1.0, 0.3]))
    grid, origin, esdf = _scene(wall_x=0.6)
    T = _camera_pose(0.0, 0.0, 0.0)  # facing +x, wall 0.6m ahead of the camera
    v, w, poses, _, guard = planner.step(grid, origin, T, 0.0, np.array([3.0, 0.0]), esdf)
    assert guard and v == 0.0 and np.isclose(w, 0.3 * W_MAX)
    assert np.allclose(poses[:, :2], poses[0, :2])  # turning in place


def test_reset_clears_memory_and_trail():
    pol = FakePolicy([0.5, 0.0])
    planner = RLPlanner(pol)
    grid, origin, esdf = _scene()
    for k in range(5):
        planner.step(grid, origin, _camera_pose(0.1 * k, 0.0, 0.0), 0.15 * k, np.array([3.0, 0.0]), esdf)
    assert len(planner.hist) == 5 and planner.vel[0] > 0.3 and pol.seen[-1][0][2].max() > 0
    planner.reset()
    assert pol.resets == 2 and planner.hist == [] and not planner.vel.any() and not planner.last_action.any()

import math

import numpy as np
import pytest
import torch

from tinynav.core.planning_node import update_occupancy_grid
from tinynav.rl.bev import GRID_RES, RAY_STEP, make_bev, trail_channel
from tinynav.rl.sim import NavSim, SimConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _drive(sim, n_frames=25):
    """Feed identical depth to the sim's torch fusion and the planner's numpy fusion."""
    c = sim.cfg
    K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1.0]])
    idx = torch.tensor([0], device=DEVICE)
    g_np, o_np = sim.unrolled_grid(0)
    g_np, o_np = g_np.double().cpu().numpy(), o_np.double().cpu().numpy()
    for f in range(n_frames):
        sim.pos[0] += 0.12 * torch.stack([torch.cos(sim.yaw[0]), torch.sin(sim.yaw[0])])
        sim.yaw[0] += 0.15 * math.sin(f / 4)
        R, cam = sim.camera_pose(idx)
        depth = sim._render(idx, R, cam)
        sim._recenter(idx, cam)
        sim._fuse(idx, R, cam, depth)
        img = np.full((c.img_h, c.img_w), np.nan)
        img[::RAY_STEP, ::RAY_STEP] = depth[0].cpu().double().numpy().reshape(img[::RAY_STEP, ::RAY_STEP].shape)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R[0].cpu().double().numpy(), cam[0].cpu().double().numpy()
        g_np, o_np = update_occupancy_grid(g_np, o_np, img, T, K, GRID_RES, RAY_STEP)
    return g_np, o_np, T


@pytest.mark.parametrize("seed", [3, 7, 11, 19])
def test_sim_fusion_and_bev_match_planner(seed):
    sim = NavSim(SimConfig(num_envs=2, n_boxes=16), device=DEVICE, seed=seed)
    sim.reset()
    g_np, o_np, T = _drive(sim)
    g_t, o_t = sim.unrolled_grid(0)
    g_t, o_t = g_t.cpu().numpy(), o_t.cpu().numpy()
    assert np.allclose(o_t, o_np, atol=1e-4), (o_t, o_np)
    match = np.isclose(g_t, g_np, atol=1e-4).mean()
    assert match > 0.9995, match  # float32 ray interpolation may flip a few voxels on exact .5 ties
    assert (g_np > 0.1).sum() > 50  # the scene actually produced obstacles / ground hits

    bev_np = make_bev(g_np, o_np, GRID_RES, sim.pos[0].cpu().numpy(), float(sim.yaw[0]), ref_z=T[2, 3])
    bev_t = sim.bev(torch.tensor([0], device=DEVICE))[0].cpu().numpy()
    assert np.isclose(bev_t[:2], bev_np, atol=1e-4).mean() > 0.99  # those tie voxels can move a wall top by one cell


def test_trail_matches_numpy():
    sim = NavSim(SimConfig(num_envs=2), device=DEVICE, seed=5)
    sim.reset()
    for _ in range(60):
        sim.step(torch.tensor([[0.6, 0.5], [0.4, -0.3]], device=DEVICE))
    h = sim.hist[1].cpu().numpy()
    ok = np.isfinite(h[:, 0])
    ref = trail_channel(h[ok], np.arange(len(h))[ok], sim.pos[1].cpu().numpy(), float(sim.yaw[1]))
    got = sim.bev(torch.tensor([1], device=DEVICE))[0, 2].cpu().numpy()
    assert ref.max() > 0.9 and (ref > 0).sum() > 20  # the recent path is drawn
    assert np.allclose(got, ref, atol=1e-5)


def test_sim_step_and_reset():
    sim = NavSim(SimConfig(num_envs=8), device=DEVICE, seed=0)
    sim.reset()
    obs = sim.observe()
    assert obs["bev"].shape == (8, 3, 64, 64) and obs["vec"].shape == (8, 6)
    done_any = False
    for _ in range(40):
        obs, rew, done, info = sim.step(torch.tensor([[1.0, 0.0]], device=DEVICE).expand(8, 2))
        assert torch.isfinite(rew).all() and torch.isfinite(obs["bev"]).all()
        done_any |= bool(done.any())
    assert done_any  # driving straight for 40 steps should hit something or a border
    assert (sim.steps < 40).any()  # finished envs were reset


def test_policy_consumes_sim_obs():
    from tinynav.rl.policy import ActorCritic
    sim = NavSim(SimConfig(num_envs=4), device=DEVICE, seed=0)
    sim.reset()
    obs = sim.observe()
    net = ActorCritic().to(DEVICE)
    act = net(obs["bev"], obs["vec"])
    assert act.shape == (4, 2) and act.abs().max() <= 1
    dist, val = net.dist_value(obs["bev"], obs["vec"])
    loss = -dist.log_prob(act).sum() + val.pow(2).sum()
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())

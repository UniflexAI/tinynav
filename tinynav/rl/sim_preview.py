"""Side-by-side check that sim BEVs look like real ones, plus ring statistics.

Usage:
  uv run python -m tinynav.rl.sim_preview --real tinynav_db/rl_ref/bev_real_0716.npz --png sim_vs_real.png
"""
import argparse

import numpy as np
import torch

from tinynav.rl.bev import robot_frame_cells
from tinynav.rl.sim import NavSim, SimConfig


def ring_stats(bev):
    """Per 0.5m ring in the forward 80 deg cone: known fraction, flat|known, obstacle|known (elevation)."""
    fwd, left = robot_frame_cells()
    d, front = np.hypot(fwd, left), np.abs(np.degrees(np.arctan2(left, fwd))) < 40
    h, k = bev[:, 0], bev[:, 1] > 0
    ground, obst = np.abs(h) < 0.15, h >= 0.15
    rows = []
    for r0 in np.arange(0.5, 3.2, 0.5):
        m = front & (d >= r0) & (d < r0 + 0.5)
        kk = k[:, m]
        rows.append((r0, kk.mean(), (ground[:, m] & kk).sum() / max(kk.sum(), 1), (obst[:, m] & kk).sum() / max(kk.sum(), 1)))
    return rows


def rollout(sim, steps):
    """Drive toward the goal with a P-controller; returns BEVs and world snapshots of env 0..3."""
    bevs, snaps = [], []
    for t in range(steps):
        g = sim.goal_robot_frame()
        a = torch.stack([(g[:, 0] / 2).clamp(-0.2, 0.6), (torch.atan2(g[:, 1], g[:, 0]) * 1.5).clamp(-1, 1)], 1)
        obs, _, _, _ = sim.step(a)
        if t >= 10 and t % 5 == 0:
            bevs.append(obs["bev"].cpu().numpy())
        if t == steps - 1:
            snaps = [dict(boxes=sim.boxes[i].cpu().numpy(), pos=sim.pos[i].cpu().numpy(), yaw=float(sim.yaw[i]),
                          goal=sim.goal[i].cpu().numpy(), bev=obs["bev"][i].cpu().numpy()) for i in range(4)]
    return np.concatenate(bevs), snaps


def plot(real, snaps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ext = (-3.2, 3.2, -3.2, 3.2)
    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    for j, i in enumerate(np.linspace(10, len(real) - 1, 4).astype(int)):
        ax[0, j].imshow(real[i, 0], cmap="viridis", vmin=-0.5, vmax=1.0, extent=ext)
        ax[0, j].set_title(f"REAL frame {i}")
    for j, s in enumerate(snaps):
        a = ax[1, j]
        for b in s["boxes"]:
            if b[4] > 0:
                a.add_patch(Rectangle((b[0] - b[2], b[1] - b[3]), 2 * b[2], 2 * b[3], color=plt.cm.viridis(min(b[4], 1.5) / 1.5)))
        p, y = s["pos"], s["yaw"]
        a.plot(*p, "r^", ms=8)
        a.plot([p[0], p[0] + 3.2 * np.cos(y + 0.72)], [p[1], p[1] + 3.2 * np.sin(y + 0.72)], "r--", lw=0.8)
        a.plot([p[0], p[0] + 3.2 * np.cos(y - 0.72)], [p[1], p[1] + 3.2 * np.sin(y - 0.72)], "r--", lw=0.8)
        a.plot(*s["goal"], "m*", ms=14)
        a.set_xlim(p[0] - 4, p[0] + 4), a.set_ylim(p[1] - 4, p[1] + 4), a.set_aspect("equal")
        a.set_title(f"SIM env {j}: world (top-down)")
        ax[2, j].imshow(s["bev"][0], cmap="viridis", vmin=-0.5, vmax=1.0, extent=ext)
        ax[2, j].set_title(f"SIM env {j}: BEV elevation (heading up)")
    for a in ax[[0, 2]].ravel():
        a.plot(0, 0, "r^", ms=8)
    fig.savefig(path, dpi=70, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--png", default="sim_vs_real.png")
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    real = np.load(args.real)["bev"]
    sim = NavSim(SimConfig(num_envs=64), device="cuda" if torch.cuda.is_available() else "cpu", seed=1)
    sim.reset()
    simb, snaps = rollout(sim, args.steps)
    print("ring(m)   known  flat|known  obst|known      (real vs sim)")
    for r, s in zip(ring_stats(real), ring_stats(simb)):
        print(f"{r[0]:.1f}-{r[0] + .5:.1f}   {r[1]:.2f}/{s[1]:.2f}   {r[2]:.2f}/{s[2]:.2f}        {r[3]:.3f}/{s[3]:.3f}")
    plot(real, snaps, args.png)


if __name__ == "__main__":
    main()

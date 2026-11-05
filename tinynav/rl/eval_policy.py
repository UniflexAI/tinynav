"""Evaluate a checkpoint on the hardest curriculum level and render a GIF.

Usage:
  uv run python -m tinynav.rl.eval_policy --ckpt tinynav_db/rl_runs/ppo_v1/last.pt --gif eval.gif
"""
import argparse

import numpy as np
import torch

from tinynav.rl.policy import ActorCritic
from tinynav.rl.sim import NavSim, SimConfig
from tinynav.rl.train_ppo import CURRICULUM, set_level


class PolicyActor:
    """Runs the recurrent policy, clearing its memory for envs that just finished an episode."""
    def __init__(self, net, n):
        self.net, self.h = net, net.initial_state(n, "cuda")

    def __call__(self, obs, done):
        self.h = self.h * (~done)[:, None]
        act, self.h = self.net.act(obs["bev"], obs["vec"], self.h)
        return act


def straight_to_goal(obs, done):
    g = obs["vec"][:, :2]
    return torch.stack([torch.full_like(g[:, 0], 0.6), (torch.atan2(g[:, 1], g[:, 0]) * 1.5).clamp(-1, 1)], 1)


def evaluate(make_actor, level, envs, seed, record=0, trap_prob=0.3, narrow_prob=0.25):
    """One episode per env; returns outcome rates (all, trap-only), quality metrics and, for the first `record`
    envs, per-step frames."""
    sim = NavSim(SimConfig(num_envs=envs, trap_prob=trap_prob, narrow_prob=narrow_prob), device="cuda", seed=seed)
    set_level(sim, level)
    sim.reset()
    obs, act_fn, done = sim.observe(), make_actor(envs), torch.zeros(envs, dtype=torch.bool, device=sim.device)
    outcome = torch.full((envs,), -1, device=sim.device)  # 0 reached, 1 collided, 2 timeout
    min_clear, eff, abs_w = torch.zeros(envs, device=sim.device), torch.zeros(envs, device=sim.device), torch.zeros(envs, device=sim.device)
    frames = [[] for _ in range(record)]
    trap, narrow = sim.trap.clone(), sim.narrow.clone()
    boxes = [sim.boxes[i].cpu().numpy() for i in range(record)]
    while (outcome < 0).any():
        for i in range(record):
            if outcome[i] < 0:
                frames[i].append((sim.pos[i].cpu().numpy(), float(sim.yaw[i]), sim.goal[i].cpu().numpy(), obs["bev"][i].cpu().numpy()))
        with torch.no_grad():
            obs, _, done, info = sim.step(act_fn(obs, done))
        first = done & (outcome < 0)
        min_clear[first], eff[first], abs_w[first] = info["min_clearance"][first], info["efficiency"][first], info["mean_abs_w"][first]
        outcome[first & info["reached"]] = 0
        outcome[first & info["collided"]] = 1
        outcome[first & info["timeout"]] = 2
    rates = [(outcome == k).float().mean().item() for k in range(3)]
    trap_rates = [(outcome[trap] == k).float().mean().item() for k in range(3)]
    ok = outcome == 0
    quality = {"near_miss": (min_clear[outcome != 1] < 0.1).float().mean().item(),  # came within 10cm without colliding
               "efficiency": eff[ok].mean().item(), "mean_abs_w": abs_w[ok].mean().item(),
               "narrow_success": ok[narrow].float().mean().item() if narrow.any() else float("nan")}
    return rates, trap_rates, quality, list(zip(boxes, frames, outcome[:record].tolist()))


def render_gif(episodes, path, stride=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    imgs = []
    names = ["REACHED", "COLLIDED", "TIMEOUT"]
    for boxes, frames, out in episodes:
        path_xy = np.array([f[0] for f in frames])
        for t in list(range(0, len(frames), stride)) + [len(frames) - 1] * 6:
            pos, yaw, goal, bev = frames[t]
            fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5))
            for bx in boxes:
                if bx[4] > 0:
                    a.add_patch(Rectangle((bx[0] - bx[2], bx[1] - bx[3]), 2 * bx[2], 2 * bx[3], color=plt.cm.viridis(min(bx[4], 1.0))))
            a.plot(path_xy[:t + 1, 0], path_xy[:t + 1, 1], "r-", lw=1.5)
            a.arrow(pos[0], pos[1], 0.4 * np.cos(yaw), 0.4 * np.sin(yaw), color="r", width=0.06)
            a.plot(*goal, "m*", ms=16)
            lo, hi = np.minimum(path_xy.min(0), goal) - 2.5, np.maximum(path_xy.max(0), goal) + 2.5
            a.set_xlim(lo[0], hi[0]), a.set_ylim(lo[1], hi[1]), a.set_aspect("equal")
            a.set_title(f"world  t={t * 0.15:.1f}s" + (f"  -> {names[out]}" if t == len(frames) - 1 else ""))
            b.imshow(bev[0], cmap="viridis", vmin=-0.5, vmax=1.0, extent=(-3.2, 3.2, -3.2, 3.2))
            b.imshow(np.ma.masked_where(bev[1] > 0, bev[1]), cmap="gray", vmin=0, vmax=1, alpha=0.6, extent=(-3.2, 3.2, -3.2, 3.2))
            b.plot(0, 0, "r^", ms=10)
            b.set_title("policy input: BEV (grey = never seen)")
            fig.canvas.draw()
            imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"wrote {path} ({len(imgs)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--level", type=int, default=len(CURRICULUM) - 1)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--gif")
    ap.add_argument("--seed", type=int, default=123)  # differs from training seeds
    ap.add_argument("--trap-prob", type=float, default=0.3)
    ap.add_argument("--narrow-prob", type=float, default=0.25)
    args = ap.parse_args()
    net = ActorCritic().cuda().eval()
    ck = torch.load(args.ckpt, map_location="cuda")
    net.load_state_dict(ck["model"])
    print(f"checkpoint step {ck['step']:,}, trained up to level {ck['level']}; evaluating level {args.level} on {args.envs} episodes")
    for name, fn in (("policy", lambda n: PolicyActor(net, n)), ("straight-to-goal", lambda n: straight_to_goal)):
        (succ, coll, tout), (ts, tc, tt), q, eps = evaluate(fn, args.level, args.envs, args.seed, record=4 if name == "policy" and args.gif else 0,
                                                            trap_prob=args.trap_prob, narrow_prob=args.narrow_prob)
        print(f"{name:>17}: success {succ:.1%}  collision {coll:.1%}  timeout {tout:.1%}   | trap episodes: success {ts:.1%}  collision {tc:.1%}  timeout {tt:.1%}")
        print(f"{'':>17}  near-miss (<10cm) {q['near_miss']:.1%}  path efficiency {q['efficiency']:.1%}  mean |yaw rate| {q['mean_abs_w']:.3f} rad/s"
              f"  narrow-passage success {q['narrow_success']:.1%}")
        if eps:
            render_gif(eps, args.gif)


if __name__ == "__main__":
    main()

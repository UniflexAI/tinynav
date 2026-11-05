"""Render one GIF per scenario category (easy -> traps) plus per-category success rates.

Usage:
  uv run python -m tinynav.rl.showcase --ckpt tinynav_db/rl_runs/ppo_v4_finetune/best.pt --out tinynav_db/rl_runs/showcase
"""
import argparse
import json
from pathlib import Path

import torch

from tinynav.rl.eval_policy import PolicyActor, render_gif
from tinynav.rl.policy import ActorCritic
from tinynav.rl.sim import NavSim, SimConfig
from tinynav.rl.train_ppo import set_level

# name, curriculum level, overrides on top of that level
SCENARIOS = [
    ("01_open_short", 0, dict(trap_prob=0.0, density=(0.0, 0.1))),
    ("02_few_boxes", 1, dict(trap_prob=0.0, density=(0.2, 0.3))),
    ("03_scattered_boxes", 2, dict(trap_prob=0.0, density=(0.4, 0.5))),
    ("04_long_goal", 3, dict(trap_prob=0.0, density=(0.3, 0.5), goal_dist=(6.5, 8.0))),
    ("05_dense_clutter", 4, dict(trap_prob=0.0, density=(0.8, 1.0))),
    ("06_dense_long", 4, dict(trap_prob=0.0, density=(0.9, 1.0), goal_dist=(6.5, 8.0))),
    ("07_wall_short", 4, dict(trap_prob=1.0, cup_prob=0.0, trap_half=(1.2, 2.0))),
    ("08_wall_long", 4, dict(trap_prob=1.0, cup_prob=0.0, trap_half=(2.8, 3.5))),
    ("09_u_trap", 4, dict(trap_prob=1.0, cup_prob=1.0, trap_half=(1.5, 2.5))),
    ("10_u_trap_wide_dense", 4, dict(trap_prob=1.0, cup_prob=1.0, trap_half=(2.8, 3.5), density=(0.6, 1.0))),
]


def run(net, level, overrides, envs, seed, record):
    sim = NavSim(SimConfig(num_envs=envs), device="cuda", seed=seed)
    set_level(sim, level)
    for k, v in overrides.items():
        setattr(sim.cfg, k, v)
    sim.reset()
    obs, actor = sim.observe(), PolicyActor(net, envs)
    done = torch.zeros(envs, dtype=torch.bool, device=sim.device)
    outcome = torch.full((envs,), -1, device=sim.device)
    boxes = [sim.boxes[i].cpu().numpy() for i in range(record)]
    frames = [[] for _ in range(record)]
    while (outcome < 0).any():
        for i in range(record):
            if outcome[i] < 0:
                frames[i].append((sim.pos[i].cpu().numpy(), float(sim.yaw[i]), sim.goal[i].cpu().numpy(), obs["bev"][i].cpu().numpy()))
        with torch.no_grad():
            obs, _, done, info = sim.step(actor(obs, done))
        first = done & (outcome < 0)
        for k, c in (("reached", 0), ("collided", 1), ("timeout", 2)):
            outcome[first & info[k]] = c
    return outcome.cpu().tolist(), list(zip(boxes, frames, outcome[:record].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=128, help="episodes per category for the success rate")
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = ActorCritic().cuda().eval()
    net.load_state_dict(torch.load(args.ckpt, map_location="cuda")["model"])
    summary = []
    for name, level, ov in SCENARIOS:
        outcomes, eps = run(net, level, ov, args.envs, args.seed, record=1)
        rates = {k: sum(o == c for o in outcomes) / len(outcomes) for k, c in (("success", 0), ("collision", 1), ("timeout", 2))}
        boxes, frames, res = eps[0]  # the first episode of the batch, whatever its outcome
        render_gif([(boxes, frames, res)], out / f"{name}.gif", stride=2)
        summary.append({"name": name, "level": level, **rates, "shown": ["reached", "collided", "timeout"][res], "seconds": round(len(frames) * 0.15, 1)})
        print(f"{name:24s} success {rates['success']:.0%} collision {rates['collision']:.0%} timeout {rates['timeout']:.0%}  shown: {summary[-1]['shown']} ({summary[-1]['seconds']}s)", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

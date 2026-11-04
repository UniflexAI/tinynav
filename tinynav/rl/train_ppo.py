"""PPO training of the BEV navigation policy in NavSim.

Usage (inside tinynav-dev):
  uv run python -m tinynav.rl.train_ppo --run tinynav_db/rl_runs/ppo_v1 --steps 20_000_000
  tensorboard --logdir tinynav_db/rl_runs --port 6006
Curriculum: start with near goals and few boxes, level up when the last update's episodes
succeed >= 80% with <= 10% collisions.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tinynav.rl.bev import CHANNELS
from tinynav.rl.policy import ActorCritic
from tinynav.rl.sim import NavSim, SimConfig

# level -> (max goal distance m, obstacle density range)
CURRICULUM = [(3.0, (0.0, 0.2)), (4.0, (0.1, 0.4)), (5.5, (0.2, 0.6)), (7.0, (0.3, 0.8)), (8.0, (0.3, 1.0))]


def set_level(sim, level):
    goal_max, density = CURRICULUM[level]
    sim.cfg.goal_dist, sim.cfg.density = (2.0, goal_max), density


def load_weights(net, state):
    """Load a checkpoint, zero-padding the first conv if it was trained with fewer BEV channels."""
    w = state["cnn.0.weight"]
    if w.shape[1] < len(CHANNELS):
        state["cnn.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], len(CHANNELS) - w.shape[1], *w.shape[2:])], 1)
    net.load_state_dict(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--init", help="start from this checkpoint's weights (fresh optimizer, step 0)")
    ap.add_argument("--level", type=int, default=0, help="starting curriculum level")
    ap.add_argument("--init-std", type=float, help="reset the action std after --init, to explore again")
    args = ap.parse_args()
    run, dev = Path(args.run), torch.device("cuda")
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    writer = SummaryWriter(run)

    sim = NavSim(SimConfig(num_envs=args.envs), device=dev, seed=args.seed)
    level = args.level
    set_level(sim, level)
    sim.reset()
    net = ActorCritic().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    step = 0
    if args.init:
        load_weights(net, torch.load(args.init, map_location=dev)["model"])
        if args.init_std:
            net.log_std.data.fill_(float(np.log(args.init_std)))
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"])
        level, step = ck["level"], ck["step"]
        set_level(sim, level)

    N, T = args.envs, args.horizon
    buf_bev = torch.zeros((T, N, len(CHANNELS), 64, 64), dtype=torch.float16, device=dev)
    buf_vec = torch.zeros((T, N, 6), device=dev)
    buf_act, buf_logp = torch.zeros((T, N, 2), device=dev), torch.zeros((T, N), device=dev)
    buf_rew, buf_done, buf_val = torch.zeros((T, N), device=dev), torch.zeros((T, N), device=dev), torch.zeros((T, N), device=dev)
    ep_ret, ep_len = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    obs = sim.observe()
    update, t0, step0 = 0, time.time(), step
    while step < args.steps:
        stats = {"reached": 0, "collided": 0, "timeout": 0, "ret": [], "len": [], "trap_n": 0, "trap_ok": 0}
        with torch.no_grad():
            for t in range(T):
                dist, val = net.dist_value(obs["bev"], obs["vec"])
                act = dist.sample()
                buf_bev[t], buf_vec[t], buf_act[t] = obs["bev"].half(), obs["vec"], act
                buf_logp[t], buf_val[t] = dist.log_prob(act).sum(1), val
                obs, rew, done, info = sim.step(act)
                buf_rew[t], buf_done[t] = rew, done.float()
                ep_ret += rew
                ep_len += 1
                if done.any():
                    for k in ("reached", "collided", "timeout"):
                        stats[k] += int(info[k].sum())
                    stats["trap_n"] += int((done & info["trap"]).sum())
                    stats["trap_ok"] += int((info["reached"] & info["trap"]).sum())
                    stats["ret"] += ep_ret[done].tolist()
                    stats["len"] += ep_len[done].tolist()
                    ep_ret[done], ep_len[done] = 0, 0
            # GAE; a timeout is treated as terminal (small bias, keeps auto-reset simple)
            next_val = net.dist_value(obs["bev"], obs["vec"])[1]
            adv, last = torch.zeros_like(buf_rew), torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else buf_val[t + 1]
                nonterm = 1.0 - buf_done[t]
                delta = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + buf_val
        step += N * T

        b_bev, b_vec, b_act = buf_bev.flatten(0, 1), buf_vec.flatten(0, 1), buf_act.flatten(0, 1)
        b_logp, b_adv, b_ret = buf_logp.flatten(), adv.flatten(), ret.flatten()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        for _ in range(args.epochs):
            for mb in torch.randperm(N * T, device=dev).split(args.minibatch):
                dist, val = net.dist_value(b_bev[mb].float(), b_vec[mb])
                logp = dist.log_prob(b_act[mb]).sum(1)
                ratio = (logp - b_logp[mb]).exp()
                pg = -torch.min(ratio * b_adv[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * b_adv[mb]).mean()
                vf = 0.5 * ((val - b_ret[mb]) ** 2).mean()
                ent = dist.entropy().sum(1).mean()
                loss = pg + 0.5 * vf - args.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        n_ep = stats["reached"] + stats["collided"] + stats["timeout"]
        if n_ep:
            succ, coll = stats["reached"] / n_ep, stats["collided"] / n_ep
            for k, v in (("success_rate", succ), ("collision_rate", coll), ("timeout_rate", stats["timeout"] / n_ep),
                         ("mean_return", np.mean(stats["ret"])), ("mean_episode_len", np.mean(stats["len"]))):
                writer.add_scalar(f"episode/{k}", v, step)
            if stats["trap_n"]:
                writer.add_scalar("episode/trap_success_rate", stats["trap_ok"] / stats["trap_n"], step)
            if succ >= 0.8 and coll <= 0.1 and level < len(CURRICULUM) - 1:
                level += 1
                set_level(sim, level)
        writer.add_scalar("train/curriculum_level", level, step)
        writer.add_scalar("train/policy_loss", pg.item(), step)
        writer.add_scalar("train/value_loss", vf.item(), step)
        writer.add_scalar("train/entropy", ent.item(), step)
        writer.add_scalar("train/action_std", net.log_std.exp().mean().item(), step)
        writer.add_scalar("train/steps_per_sec", (step - step0) / (time.time() - t0), step)
        update += 1
        if update % 10 == 0 or step >= args.steps:
            print(f"step {step:>10,} lvl {level} eps {n_ep:4d} succ {succ if n_ep else 0:.2f} coll {coll if n_ep else 0:.2f} "
                  f"trap {stats['trap_ok'] / max(stats['trap_n'], 1):.2f} ret {np.mean(stats['ret']) if n_ep else 0:6.2f} std {net.log_std.exp().mean():.2f} {(time.time() - t0) / 60:.1f} min", flush=True)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "level": level, "step": step}, run / "last.pt")
        if update % 100 == 0:
            torch.save({"model": net.state_dict(), "level": level, "step": step}, run / f"step_{step}.pt")


if __name__ == "__main__":
    main()

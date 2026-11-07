"""Recurrent PPO training of the BEV navigation policy in NavSim.

Usage (inside tinynav-dev):
  uv run python -m tinynav.rl.train_ppo --run tinynav_db/rl_runs/ppo_v3_gru --steps 30_000_000
  tensorboard --logdir tinynav_db/rl_runs --port 6006
Curriculum: level up when episodes over the last 10 updates succeed >= 80% with <= 10% collisions,
and only after LEVEL_COOLDOWN updates at the current level: episodes run up to 400 steps, so
right after a level change the finished episodes still come from the easier level.
Each env's rollout is one GRU sequence; minibatches split by env so sequences stay whole.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tinynav.rl.bev import CHANNELS
from tinynav.rl.policy import VEC_DIM, ActorCritic
from tinynav.rl.sim import PRIV_DIM, NavSim, SimConfig

# level -> (max goal distance m, obstacle density range, trap wall half-length range, U-cup share,
#           doorway width range m, share of non-trap episodes with a doorway)
# Doorways start wide and shrink: at 0.8-1.2m from the start the policy almost never tried a door
# (a clipped frame ends the episode at -10) and learned to walk around the wall instead.
CURRICULUM = [
    (3.0, (0.0, 0.2), (0.6, 1.0), 0.0, (1.4, 2.0), 0.5),
    (4.0, (0.1, 0.4), (0.8, 1.5), 0.0, (1.2, 1.7), 0.5),
    (5.5, (0.2, 0.6), (1.0, 2.0), 0.2, (1.0, 1.4), 0.45),
    (7.0, (0.3, 0.8), (1.2, 2.8), 0.35, (0.9, 1.3), 0.4),
    (8.0, (0.3, 1.0), (1.5, 3.5), 0.5, (0.8, 1.2), 0.35),
]
NARROW_PROMOTE = 0.6  # doorway success needed to level up, on top of overall success / collisions


LEVEL_COOLDOWN = 20


class Demos:
    """Human-driven demonstrations (tinynav.rl.demos) cut into GRU sequences for a behaviour-cloning loss.
    Only frames where the driver was moving and the RL footprint stayed clear count; narrow passages
    weigh narrow_weight times more. The first `burn` frames of a sequence only warm up the memory."""
    def __init__(self, pattern, seq, dev, narrow_weight=3.0, burn=16):
        bev, vec, act, wt = [], [], [], []
        for path in sorted(glob.glob(pattern)):
            d = dict(np.load(path))  # npz members decompress on every access; read each once
            w = (d["moving"] & d["safe"]) * (1 + (narrow_weight - 1) * d["narrow"])
            for s0 in range(0, len(w) - seq + 1, seq // 2):
                sl = slice(s0, s0 + seq)
                bev.append(d["bev"][sl]), vec.append(d["vec"][sl]), act.append(d["action"][sl]), wt.append(w[sl])
        if not bev:
            raise SystemExit(f"no demonstrations matched {pattern}")
        # kept in (pinned) host memory, ~1GB for 20k frames; only the sampled batch goes to the GPU
        self.bev = torch.tensor(np.stack(bev), dtype=torch.float16).pin_memory()  # [S,T,3,64,64]
        self.vec, self.act = torch.tensor(np.stack(vec)), torch.tensor(np.stack(act))
        self.wt = torch.tensor(np.stack(wt), dtype=torch.float32)
        self.wt[:, :burn] = 0
        self.burn, self.dev = burn, dev
        print(f"demos: {len(bev)} sequences of {seq} frames, weighted frames {self.wt.gt(0).sum().item():,} "
              f"(narrow {self.wt.gt(1).sum().item():,})", flush=True)

    def loss(self, net, batch):
        """Weighted squared error between the actor's mean action and the driver's, through the GRU."""
        i = torch.randint(len(self.bev), (batch,))
        bev, vec, act, wt = (x[i].to(self.dev, non_blocking=True) for x in (self.bev, self.vec, self.act, self.wt))  # [B,T,...]
        B, T = wt.shape
        feats = net.encode(bev.flatten(0, 1).float(), vec.flatten(0, 1)).view(B, T, -1)
        h, means = net.initial_state(B, self.dev), []
        for t in range(T):
            h = net.gru(feats[:, t], h)
            means.append(net.actor(h))
        err = ((torch.stack(means, 1) - act) ** 2).sum(-1)
        return (err * wt).sum() / wt.sum().clamp(min=1)


def set_level(sim, level):
    goal_max, density, trap_half, cup, door, narrow = CURRICULUM[level]
    sim.cfg.goal_dist, sim.cfg.density, sim.cfg.trap_half, sim.cfg.cup_prob = (2.0, goal_max), density, trap_half, cup
    sim.cfg.door_width, sim.cfg.narrow_prob = door, narrow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrot-distance", type=float, default=0.0)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=30_000_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)  # detours take 100+ steps
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-end", type=float, help="linearly anneal the learning rate to this by --steps")
    ap.add_argument("--resume")
    ap.add_argument("--init", help="start from this checkpoint's weights (fresh optimizer, step 0)")
    ap.add_argument("--level", type=int, default=0, help="starting curriculum level")
    ap.add_argument("--eval-every", type=int, default=100, help="updates between fixed-scene evals that keep best.pt (success - 0.5 * near-miss)")
    ap.add_argument("--demos", help="glob of tinynav.rl.demos npz files for a behaviour-cloning loss")
    ap.add_argument("--bc-coef", type=float, default=1.0, help="BC loss weight at the start, annealed to --bc-coef-end")
    ap.add_argument("--bc-coef-end", type=float, default=0.1)
    ap.add_argument("--bc-batch", type=int, default=16, help="demo sequences per PPO minibatch")
    args = ap.parse_args()
    run, dev = Path(args.run), torch.device("cuda")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=2))
    torch.manual_seed(args.seed)
    writer = SummaryWriter(run)

    sim = NavSim(SimConfig(num_envs=args.envs, carrot_distance=args.carrot_distance), device=dev, seed=args.seed)
    demos = Demos(args.demos, args.horizon, dev) if args.demos else None
    net = ActorCritic().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    level, step = args.level, 0
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"])
        level, step = ck["level"], ck["step"]
    set_level(sim, level)
    sim.reset()

    N, T = args.envs, args.horizon
    buf = {"bev": torch.zeros((T, N, len(CHANNELS), 64, 64), dtype=torch.float16, device=dev),
           "vec": torch.zeros((T, N, VEC_DIM), device=dev), "priv": torch.zeros((T, N, PRIV_DIM), device=dev),
           "act": torch.zeros((T, N, 2), device=dev), "logp": torch.zeros((T, N), device=dev),
           "val": torch.zeros((T, N), device=dev), "rew": torch.zeros((T, N), device=dev), "done": torch.zeros((T, N), device=dev)}
    ep_ret, ep_len = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    obs, priv = sim.observe(), sim.privileged()
    h = net.initial_state(N, dev)
    update, t0, step0 = 0, time.time(), step
    window, level_since = [], 0  # per-update (reached, collided, episodes)
    best_eval = -1.0
    while step < args.steps:
        stats = {"reached": 0, "collided": 0, "timeout": 0, "ret": [], "len": [], "trap_n": 0, "trap_ok": 0, "narrow_n": 0, "narrow_ok": 0, "guard": 0.0,
                 "clear": [], "eff": [], "w": []}
        h0 = h.clone()
        with torch.no_grad():
            for t in range(T):
                h = net.gru(net.encode(obs["bev"], obs["vec"]), h)
                dist, val = net.heads(h, priv)
                act = dist.sample()
                buf["bev"][t], buf["vec"][t], buf["priv"][t], buf["act"][t] = obs["bev"].half(), obs["vec"], priv, act
                buf["logp"][t], buf["val"][t] = dist.log_prob(act).sum(1), val
                obs, rew, done, info = sim.step(act)
                rew = rew + args.gamma * val * info["timeout"]  # bootstrap time-outs: cut off, not failed
                priv = sim.privileged()
                buf["rew"][t], buf["done"][t] = rew, done.float()
                stats["guard"] += info["guard"].float().mean().item() / T
                h = h * (~done)[:, None]
                ep_ret += rew
                ep_len += 1
                if done.any():
                    for k in ("reached", "collided", "timeout"):
                        stats[k] += int(info[k].sum())
                    stats["trap_n"] += int((done & info["trap"]).sum())
                    stats["trap_ok"] += int((info["reached"] & info["trap"]).sum())
                    stats["narrow_n"] += int((done & info["narrow"]).sum())
                    stats["narrow_ok"] += int((info["reached"] & info["narrow"]).sum())
                    stats["clear"] += info["min_clearance"][done & ~info["collided"]].tolist()
                    stats["eff"] += info["efficiency"][info["reached"]].tolist()
                    stats["w"] += info["mean_abs_w"][done].tolist()
                    stats["ret"] += ep_ret[done].tolist()
                    stats["len"] += ep_len[done].tolist()
                    ep_ret[done], ep_len[done] = 0, 0
            # GAE; time-outs end the sequence but were bootstrapped with V(s_t) above
            next_val = net.heads(net.gru(net.encode(obs["bev"], obs["vec"]), h), priv)[1]
            adv, last = torch.zeros_like(buf["rew"]), torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else buf["val"][t + 1]
                nonterm = 1.0 - buf["done"][t]
                delta = buf["rew"][t] + args.gamma * nv * nonterm - buf["val"][t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + buf["val"]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        step += N * T

        if args.lr_end is not None:
            for g in opt.param_groups:
                g["lr"] = args.lr + (args.lr_end - args.lr) * min(1.0, (step - step0) / max(args.steps - step0, 1))
        envs_per_mb = max(1, args.minibatch // T)
        for _ in range(args.epochs):
            for e in torch.randperm(N, device=dev).split(envs_per_mb):
                B = len(e)
                feats = net.encode(buf["bev"][:, e].flatten(0, 1).float(), buf["vec"][:, e].flatten(0, 1)).view(T, B, -1)
                hs, hh = [], h0[e]
                for t in range(T):
                    hh = net.gru(feats[t], hh)
                    hs.append(hh)
                    hh = hh * (1 - buf["done"][t, e])[:, None]
                dist, val = net.heads(torch.stack(hs).flatten(0, 1), buf["priv"][:, e].flatten(0, 1))
                logp = dist.log_prob(buf["act"][:, e].flatten(0, 1)).sum(1)
                a_mb, r_mb = adv[:, e].flatten(), ret[:, e].flatten()
                ratio = (logp - buf["logp"][:, e].flatten()).exp()
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb).mean()
                vf = 0.5 * ((val - r_mb) ** 2).mean()
                ent = dist.entropy().sum(1).mean()
                loss = pg + 0.5 * vf - args.ent * ent
                if demos is not None:
                    frac = min(1.0, (step - step0) / max(args.steps - step0, 1))
                    bc = demos.loss(net, args.bc_batch)
                    loss = loss + (args.bc_coef + (args.bc_coef_end - args.bc_coef) * frac) * bc
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        n_ep = stats["reached"] + stats["collided"] + stats["timeout"]
        succ = stats["reached"] / max(n_ep, 1)
        coll = stats["collided"] / max(n_ep, 1)
        trap = stats["trap_ok"] / max(stats["trap_n"], 1)
        if n_ep:
            for k, v in (("success_rate", succ), ("collision_rate", coll), ("timeout_rate", stats["timeout"] / n_ep),
                         ("mean_return", np.mean(stats["ret"])), ("mean_episode_len", np.mean(stats["len"]))):
                writer.add_scalar(f"episode/{k}", v, step)
            if stats["trap_n"]:
                writer.add_scalar("episode/trap_success_rate", trap, step)
            if stats["narrow_n"]:
                writer.add_scalar("episode/narrow_success_rate", stats["narrow_ok"] / stats["narrow_n"], step)
            if stats["clear"]:
                writer.add_scalar("episode/near_miss_rate", float(np.mean(np.array(stats["clear"]) < 0.1)), step)
            if stats["eff"]:
                writer.add_scalar("episode/path_efficiency", float(np.mean(stats["eff"])), step)
            writer.add_scalar("episode/mean_abs_yaw_rate", float(np.mean(stats["w"])), step)
        writer.add_scalar("train/guard_intervention", stats["guard"], step)
        window = (window + [(stats["reached"], stats["collided"], n_ep, stats["narrow_ok"], stats["narrow_n"])])[-10:]
        w_ep, w_nar = sum(w[2] for w in window), sum(w[4] for w in window)
        if (update - level_since >= LEVEL_COOLDOWN and w_ep and sum(w[0] for w in window) / w_ep >= 0.8
                and sum(w[1] for w in window) / w_ep <= 0.1 and w_nar and sum(w[3] for w in window) / w_nar >= NARROW_PROMOTE
                and level < len(CURRICULUM) - 1):
            level, level_since = level + 1, update
            set_level(sim, level)
        if update % args.eval_every == 0 and level == len(CURRICULUM) - 1:
            from tinynav.rl.eval_policy import PolicyActor, evaluate  # local: eval_policy imports this module
            (e_succ, e_coll, _), (e_trap, _, _), q, _ = evaluate(lambda n: PolicyActor(net, n), level, 256, seed=999, carrot_distance=args.carrot_distance)
            for k, v in (("success_rate", e_succ), ("collision_rate", e_coll), ("trap_success_rate", e_trap), *q.items()):
                writer.add_scalar(f"eval/{k}", v, step)
            print(f"eval @ {step:,}: success {e_succ:.3f} collision {e_coll:.3f} trap {e_trap:.3f} near_miss {q['near_miss']:.3f} "
                  f"efficiency {q['efficiency']:.3f} |w| {q['mean_abs_w']:.3f}", flush=True)
            score = e_succ - 0.5 * q["near_miss"]  # success alone would keep a policy that still scrapes walls
            if score > best_eval:
                best_eval = score
                torch.save({"model": net.state_dict(), "level": level, "step": step, "eval": (e_succ, e_coll, e_trap), "config": vars(args)}, run / "best.pt")
        if demos is not None:
            writer.add_scalar("train/bc_loss", bc.item(), step)
        for k, v in (("curriculum_level", level), ("lr", opt.param_groups[0]["lr"]), ("policy_loss", pg.item()), ("value_loss", vf.item()),
                     ("entropy", ent.item()), ("action_std", net.log_std.exp().mean().item()),
                     ("steps_per_sec", (step - step0) / (time.time() - t0))):
            writer.add_scalar(f"train/{k}", v, step)
        update += 1
        if update % 10 == 0 or step >= args.steps:
            print(f"step {step:>10,} lvl {level} eps {n_ep:4d} succ {succ:.2f} coll {coll:.2f} trap {trap:.2f} "
                  f"ret {np.mean(stats['ret']) if n_ep else 0:6.2f} std {net.log_std.exp().mean():.2f} {(time.time() - t0) / 60:.1f} min", flush=True)
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "level": level, "step": step, "config": vars(args)}, run / "last.pt")
        if update % 100 == 0:
            torch.save({"model": net.state_dict(), "level": level, "step": step}, run / f"step_{step}.pt")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import torch

from tinynav.rl.policy import ActorCritic
from tinynav.rl.sim import NavSim, SimConfig
from tinynav.rl.train_ppo import set_level

OUT = Path('tinynav_db/rl_runs/ppo_v13_carrot/media')
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {'v7': 'ppo_v7_bigbody', 'v12': 'ppo_v12_guarded', 'v13': 'ppo_v13_carrot'}
# Preselected seeds, retained regardless of which policy wins.
CASES = [
    ('door_fixed', 31, 0.0, dict(trap_prob=0.0, narrow_prob=1.0, density=(0.1, 0.2), goal_dist=(6.0, 8.0))),
    ('door_moving', 31, 2.2, dict(trap_prob=0.0, narrow_prob=1.0, density=(0.1, 0.2), goal_dist=(6.0, 8.0))),
    ('wall_moving', 32, 2.2, dict(trap_prob=1.0, cup_prob=0.0, density=(0.1, 0.2), goal_dist=(6.0, 8.0))),
    ('clutter_moving', 33, 2.2, dict(trap_prob=0.0, narrow_prob=0.0, density=(0.8, 1.0), goal_dist=(6.0, 8.0))),
]
summary = []
for case, seed, carrot, overrides in CASES:
    reference = None
    for name, run in MODELS.items():
        net = ActorCritic().cuda().eval()
        ck = torch.load(Path('tinynav_db/rl_runs') / run / 'best.pt', map_location='cuda')
        net.load_state_dict(ck['model'])
        sim = NavSim(SimConfig(num_envs=1, carrot_distance=carrot), seed=seed)
        set_level(sim, 4)
        for k, v in overrides.items():
            setattr(sim.cfg, k, v)
        sim.reset()
        boxes, goal = sim.boxes[0].cpu().numpy().copy(), sim.goal[0].cpu().numpy().copy()
        initial = (boxes.copy(), goal.copy(), sim.pos[0].cpu().numpy().copy(), sim.yaw.cpu().numpy().copy())
        if reference is None:
            reference = initial
        else:
            assert all(np.array_equal(a, b) for a, b in zip(reference, initial)), 'scene mismatch'
        obs, h = sim.observe(), net.initial_state(1, 'cuda')
        poses, targets, bevs, commands, guards = [], [], [], [], []
        with torch.no_grad():
            for tick in range(sim.cfg.max_steps):
                poses.append(np.r_[sim.pos[0].cpu().numpy(), sim.yaw[0].item()])
                targets.append(sim.policy_goal()[0].cpu().numpy().copy())
                bevs.append(obs['bev'][0].cpu().numpy().astype(np.float16))
                act, h = net.act(obs['bev'], obs['vec'], h)
                commands.append(act[0].cpu().numpy().copy())
                obs, _, done, info = sim.step(act)
                guards.append(bool(info['guard'][0]))
                if done[0]:
                    break
        result = next(k for k in ('reached', 'collided', 'timeout') if info[k][0])
        row = dict(case=case, model=name, seed=seed, carrot=carrot, checkpoint_step=ck['step'],
                   result=result, seconds=len(poses)*sim.cfg.dt, efficiency=float(info['efficiency'][0]),
                   guard_fraction=float(np.mean(guards)), narrow=bool(info['narrow'][0]))
        np.savez_compressed(OUT / f'{case}_{name}.npz', boxes=boxes, goal=goal, poses=poses,
                            targets=targets, bev=bevs, actions=commands, guards=guards)
        summary.append(row)
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(row), flush=True)

import json
from pathlib import Path
import torch
from tinynav.rl.policy import ActorCritic
from tinynav.rl.eval_policy import evaluate, PolicyActor

root = Path("tinynav_db/rl_runs")
results = []
for name in ("ppo_v7_bigbody", "ppo_v12_guarded", "ppo_v13_carrot"):
    net = ActorCritic().cuda().eval()
    net.load_state_dict(torch.load(root / name / "best.pt", map_location="cuda")["model"])
    for carrot in (0.0, 2.2):
        for seed in (2024, 2025):
            with torch.no_grad():
                rates, trap, quality, _ = evaluate(lambda n: PolicyActor(net, n), 4, 256,
                    seed=seed, narrow_prob=0.35, carrot_distance=carrot)
            row = dict(model=name, carrot=carrot, seed=seed, rates=rates, trap=trap, quality=quality)
            results.append(row)
            print(json.dumps(row), flush=True)
            (root / "ppo_v13_carrot" / "comparison.json").write_text(json.dumps(results, indent=2))

"""Export the deployment path of the policy (ActorCritic.act) to ONNX for TensorRT.

The GRU memory is an explicit input/output: feed h_out back as h on the next planner tick and
zero it when a new goal starts. Inputs bev [1,3,64,64], vec [1,6], h [1,256] -> action [1,2], h_out.
Usage:
  uv run python -m tinynav.rl.export_onnx --ckpt tinynav_db/rl_runs/ppo_v12_guarded/best.pt --out tinynav/models/rl_nav_policy_v12.onnx
  cd tinynav/models && make rl_nav
"""
import argparse

import torch

from tinynav.rl.bev import BEV_SIZE, CHANNELS
from tinynav.rl.policy import HIDDEN, VEC_DIM, ActorCritic


class Deploy(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, bev, vec, h):
        return self.net.act(bev, vec, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    net = ActorCritic().eval()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    inputs = (torch.zeros(1, len(CHANNELS), BEV_SIZE, BEV_SIZE), torch.zeros(1, VEC_DIM), torch.zeros(1, HIDDEN))
    torch.onnx.export(Deploy(net), inputs, args.out, input_names=["bev", "vec", "h"], output_names=["action", "h_out"],
                      opset_version=17, dynamo=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

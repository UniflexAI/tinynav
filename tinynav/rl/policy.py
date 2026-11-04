"""Recurrent actor-critic for local navigation.

BEV -> CNN, concat goal/velocity -> GRU memory -> actor. The GRU lets the policy commit to a
detour instead of re-deciding left vs right every tick. The critic additionally sees
privileged sim-only features (geodesic distance), so it is never needed on the robot.
"""
import torch
import torch.nn as nn

from tinynav.rl.bev import BEV_SIZE, CHANNELS
from tinynav.rl.sim import PRIV_DIM

VEC_DIM = 6  # goal (fwd, left)/5, velocity normalized, last action
HIDDEN = 256


class ActorCritic(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(len(CHANNELS), 32, 5, stride=2), nn.ELU(),  # 64 -> 30
            nn.Conv2d(32, 64, 3, stride=2), nn.ELU(),             # -> 14
            nn.Conv2d(64, 64, 3, stride=2), nn.ELU(),             # -> 6
            nn.Flatten())
        n = self.cnn(torch.zeros(1, len(CHANNELS), BEV_SIZE, BEV_SIZE)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(n + VEC_DIM, hidden), nn.ELU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2))
        self.critic = nn.Sequential(nn.Linear(hidden + PRIV_DIM, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def encode(self, bev, vec):
        return self.trunk(torch.cat([self.cnn(bev), vec], 1))

    def initial_state(self, n, device):
        return torch.zeros((n, self.gru.hidden_size), device=device)

    def act(self, bev, vec, h):
        """Deployment: deterministic action in [-1, 1] and the next memory state."""
        h = self.gru(self.encode(bev, vec), h)
        return self.actor(h).clamp(-1, 1), h

    def heads(self, h, priv):
        return torch.distributions.Normal(self.actor(h), self.log_std.exp()), self.critic(torch.cat([h, priv], 1)).squeeze(1)

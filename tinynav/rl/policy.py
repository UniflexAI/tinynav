"""Actor-critic for local navigation: BEV -> CNN, concat goal/velocity -> MLP heads."""
import torch
import torch.nn as nn

from tinynav.rl.bev import BEV_SIZE, CHANNELS

VEC_DIM = 6  # goal (fwd, left)/5, velocity normalized, last action


class ActorCritic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(len(CHANNELS), 32, 5, stride=2), nn.ELU(),  # 64 -> 30
            nn.Conv2d(32, 64, 3, stride=2), nn.ELU(),             # -> 14
            nn.Conv2d(64, 64, 3, stride=2), nn.ELU(),             # -> 6
            nn.Flatten())
        n = self.cnn(torch.zeros(1, len(CHANNELS), BEV_SIZE, BEV_SIZE)).shape[1]
        self.trunk = nn.Sequential(nn.Linear(n + VEC_DIM, hidden), nn.ELU())
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 2))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def features(self, bev, vec):
        return self.trunk(torch.cat([self.cnn(bev), vec], 1))

    def forward(self, bev, vec):
        """Deterministic action in [-1, 1] for deployment."""
        return self.actor(self.features(bev, vec)).clamp(-1, 1)

    def dist_value(self, bev, vec):
        f = self.features(bev, vec)
        return torch.distributions.Normal(self.actor(f), self.log_std.exp()), self.critic(f).squeeze(1)

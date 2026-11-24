"""Frozen Spark semantic traversal networks and their live BEV projection."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FWD, LEFT = np.meshgrid(np.linspace(2.3,-2.3,24,dtype=np.float32),
                        np.linspace(2.3,-2.3,24,dtype=np.float32),indexing="ij")

def block(in_channels, out_channels, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, dilation, dilation=dilation),
                         nn.GroupNorm(4, out_channels), nn.ReLU())


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = block(1, 16)
        self.body = nn.Sequential(block(16,24,2), block(24,32,2), block(32,32,dilation=2), block(32,32,dilation=4))
        self.out = nn.Sequential(block(48,24), nn.Conv2d(24,3,1))

    def forward(self, x):
        shallow = self.first(x)
        deep = F.interpolate(self.body(shallow),size=shallow.shape[-2:],mode='bilinear',align_corners=False)
        return self.out(torch.cat((shallow,deep),dim=1))


class PPO(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(4,16,5,2),nn.ELU(),nn.Conv2d(16,24,3,2),nn.ELU(),nn.Flatten())
        self.trunk=nn.Sequential(nn.Linear(384+6,96),nn.ELU(),nn.Linear(96,64),nn.ELU())
        self.actor=nn.Linear(64,2);self.critic=nn.Linear(64,1)
        self.logstd=nn.Parameter(torch.full((2,),-.95))
    def forward(self,x,v):
        z=self.trunk(torch.cat((self.cnn(x),v),1))
        return self.actor(z).tanh(),self.critic(z)[:,0]


def preview(bev):
    score=bev[:,3]
    occ=bev[:,0]>.5
    f=torch.as_tensor(FWD,device=bev.device)
    l=torch.as_tensor(LEFT,device=bev.device)
    rows=[]
    for slope in (-.48,0.,.48):
        patch=(f>.70)&(f<2.2)&((l-slope*f).abs()<.38)&~occ
        seen=patch&(score.abs()>1e-5)
        count=seen.sum((1,2)).float()
        mean=(score*seen).sum((1,2))/count.clamp_min(1)
        rows.extend((mean,count/patch.sum((1,2)).clamp_min(1)))
    return torch.stack(rows,1)
class PreviewPPO(PPO):
    def __init__(self):
        super().__init__()
        prev=self.trunk[0]
        self.trunk[0]=nn.Linear(prev.in_features+6,96)
    def forward(self,x,v):
        polar=preview(x)
        geom=x.clone();geom[:,3]=0.
        z=self.trunk(torch.cat((self.cnn(geom),v,polar),1))
        return self.actor(z).tanh(),self.critic(z)[:,0]


class TerminalPolicy(PreviewPPO):
    """A learned terminal action head, blended in by distance to the target."""
    def __init__(self, source=None, polar=False):
        super().__init__()
        if source:
            self.load_state_dict(torch.load(source, weights_only=True))
        self.requires_grad_(False)
        self.polar = polar
        self.terminal_actor = nn.Sequential(nn.Linear(15 if polar else 12, 48), nn.Tanh(),
                                            nn.Linear(48, 32), nn.Tanh(), nn.Linear(32, 2))
        self.critic.requires_grad_(True)

    def terminal_action(self, bev, vector):
        features = [vector, preview(bev)]
        if self.polar:
            distance = 4 * torch.linalg.vector_norm(vector[:, :2], dim=1)
            angle = torch.atan2(vector[:, 1], vector[:, 0])
            features.append(torch.stack((distance, angle / torch.pi, (distance - .205) * 5), dim=1))
        return self.terminal_actor(torch.cat(features, dim=1)).tanh()

    def forward(self, bev, vector):
        action, value = super().forward(bev, vector)
        distance = 4 * torch.linalg.vector_norm(vector[:, :2], dim=1, keepdim=True)
        weight = ((.92 - distance) / .28).clamp(0, 1)
        correction = self.terminal_action(bev, vector)
        return torch.lerp(action, correction, weight), value




def blend_score_action(scored, neutral, evidence):
    """Keep geometry as a prior and scale the score residual by observed contrast."""
    means = evidence[..., ::2]
    coverage = evidence[..., 1::2]
    paired = coverage >= .12
    high = torch.where(paired, means, -torch.inf).amax(dim=-1)
    low = torch.where(paired, means, torch.inf).amin(dim=-1)
    contrast = torch.where(paired.sum(dim=-1) >= 2, (high-low).clamp(0., 1.), 0.)
    # When all visible directions have similar scores, retain a small semantic
    # contribution while limiting response to a common brightness/classifier bias.
    weight = .15 + .35 * contrast
    return torch.lerp(neutral, scored, weight[..., None]), weight


class ScoreCommit(nn.Module):
    """Require a consistent left/right contrast before committing during approach."""
    def __init__(self, directory, max_semantic_ticks=24):
        super().__init__()
        self.prior = TerminalPolicy(polar=True)
        self.prior.load_state_dict(torch.load(directory / "synthetic-preview-terminal-polar-ppo.pt", map_location="cpu", weights_only=True))
        self.scored = TerminalPolicy(polar=True)
        self.scored.load_state_dict(torch.load(directory / "synthetic-preview-office-replay-ppo.pt", map_location="cpu", weights_only=True))
        self.max_semantic_ticks = max_semantic_ticks
        self.reset()

    def reset(self):
        self.tick = 0
        self.streak = None
        self.committed = None

    def forward(self, bev, vector):
        evidence = preview(bev)
        if self.streak is None:
            self.streak = torch.zeros(len(bev), device=bev.device)
            self.committed = torch.zeros(len(bev), dtype=torch.bool, device=bev.device)
        if self.tick < 9:
            left, right = evidence[:,4], evidence[:,0]
            high, low = torch.maximum(left,right), torch.minimum(left,right)
            established = ((high >= .23) & (low <= -.12) & (high-low >= .47)
                           & (evidence[:,[1,5]].min(1).values >= .12))
            sign = torch.sign(left-right)
            self.streak = torch.where(established,
                torch.where(torch.sign(self.streak)==sign,self.streak+sign,sign),
                torch.zeros_like(self.streak))
            self.committed |= self.streak.abs() >= 3
        self.tick += 1
        null_bev = bev.clone()
        null_bev[:,3] = 0
        base, value = self.prior(null_bev,vector)
        cautious, _ = self.prior(bev,vector)
        cautious, _ = blend_score_action(cautious,base,evidence)
        action, learned_value = self.scored(bev, vector)
        use_scores = self.committed
        if self.max_semantic_ticks is not None and self.tick > self.max_semantic_ticks:
            use_scores = torch.zeros_like(use_scores)
        return torch.where(use_scores[:,None],action,cautious), torch.where(use_scores,learned_value,value)


class LiveSemanticPolicy:
    def __init__(self):
        directory = Path(__file__).resolve().parents[1] / "models" / "semantic_bev"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ScoreCommit(directory).eval().to(self.device)
        self.student = Student().eval().to(self.device)
        self.student.load_state_dict(torch.load(directory / "r31.pt", map_location="cpu", weights_only=True))
        self.reset()

    def reset(self):
        self.actor.reset()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.bev = None

    def infer_frame(self, infra, depth, K, camera, center, yaw, goal, velocity, esdf, origin, resolution):
        if infra.ndim != 2 or infra.shape != depth.shape or infra.dtype != np.uint8:
            raise ValueError("Expected same-size mono8 Infra1 and depth")
        c, s = np.cos(yaw), np.sin(yaw)
        x = center[0] + c * FWD - s * LEFT
        y = center[1] + s * FWD + c * LEFT
        i, j = np.floor((np.stack((x, y))-origin[:2,None,None])/resolution).astype(int)
        inside = (i>=0)&(j>=0)&(i<esdf.shape[0])&(j<esdf.shape[1])
        blocked=np.ones(FWD.shape,bool)
        blocked[inside]=esdf[i[inside],j[inside]]<=0
        clearance=np.zeros_like(FWD)
        clearance[inside]=np.clip(esdf[i[inside],j[inside]]/.8,0,1)
        with torch.inference_mode():
            image=torch.as_tensor(np.ascontiguousarray(infra),device=self.device)[None,None].float()/255
            p=self.student(image).softmax(1)[0].cpu().numpy()
        valid=(p.max(0)>=.7)&np.isfinite(depth)&(depth>.15)&(depth<8)
        v,u=np.nonzero(valid)
        z=depth[v,u]
        optical=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
        xy=(optical@camera[:3,:3].T+camera[:3,3])[:,:2]-center
        forward=xy[:,0]*c+xy[:,1]*s
        lateral=-xy[:,0]*s+xy[:,1]*c
        i,j=np.floor((2.3-np.stack((forward,lateral)))/.2+.5).astype(int)
        inside=(i>=0)&(j>=0)&(i<24)&(j<24)
        scores=np.full(FWD.shape,np.inf,np.float32)
        np.minimum.at(scores,(i[inside],j[inside]),(.3*p[1,v,u]+p[2,v,u])[inside])
        scores[~np.isfinite(scores)]=.5
        scores[blocked]=0
        if not inside.any():
            raise ValueError("Infra1 projection did not observe any BEV cells")
        self.bev=np.stack((blocked,clearance,FWD/2.3,2*scores-1)).astype(np.float32)
        dx,dy=goal-center
        vector=np.asarray([(c*dx+s*dy)/4,(-s*dx+c*dy)/4,
                           velocity[0]/.55,velocity[1]/.9,*self.last_action],dtype=np.float32)
        with torch.inference_mode():
            action=self.actor(torch.as_tensor(self.bev,device=self.device)[None],
                              torch.as_tensor(vector,device=self.device)[None])[0][0].cpu().numpy()
        if not np.isfinite(action).all():
            raise ValueError("Semantic policy output is not finite")
        self.last_action=action
        speed=(float(action[0])+1)*.275
        return (speed if speed>=.05 else 0.,float(np.clip(action[1]*.9,-.75,.75)))

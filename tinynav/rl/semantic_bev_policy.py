"""Online semantic BEV using TensorRT engines built on the target."""
import asyncio
import os
from pathlib import Path

import numpy as np
import tensorrt as trt

from tinynav.core.models_trt import TRTBase

FWD, LEFT = np.meshgrid(np.linspace(2.3,-2.3,24,dtype=np.float32),
                        np.linspace(2.3,-2.3,24,dtype=np.float32),indexing="ij")

class SemanticEngine(TRTBase):
    def allocate_buffers(self):
        for index in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
                shape=self._get_static_shape(name)
                if not self.context.set_input_shape(name,shape):
                    raise ValueError(f"Could not set TensorRT shape for {name}: {shape}")
        return super().allocate_buffers()

    async def infer(self, **inputs):
        for item in self.inputs:
            value=inputs[item["name"]]
            if value.shape!=tuple(item["shape"]):
                raise ValueError(f"TensorRT {item['name']} expects {item['shape']}, got {value.shape}")
            np.copyto(item["host"],value)
        return await self.run_graph()


class LiveSemanticPolicy:
    def __init__(self):
        directory = Path(__file__).resolve().parents[1] / "models" / "semantic_bev"
        self.device = "tensorrt"
        self.record_input = False
        self.allow_neutral_projection = os.environ.get("TINYNAV_SEMANTIC_NEUTRAL_FALLBACK") == "1"
        self.semantic_projection_missing = False
        self.loop = asyncio.new_event_loop()
        self.student = SemanticEngine(directory / "r31-infra1.plan")
        self.actor = SemanticEngine(directory / "semantic-bev-ppo.plan")
        self.reset()

    def reset(self):
        self.last_action = np.zeros(2, dtype=np.float32)
        self.bev = None
        self.replay_input = None
        self.tick = 0
        self.streak = 0
        self.committed = False

    def choose(self, results, *, neutral_projection=False):
        if neutral_projection:
            self.streak = 0
            self.tick += 1
            return results["cautious"][0]
        e=results["evidence"][0]
        if self.tick<9:
            right,left=e[0],e[4]
            established=(max(left,right)>=.23 and min(left,right)<=-.12
                         and abs(left-right)>=.47 and min(e[1],e[5])>=.12)
            sign=int(np.sign(left-right))
            self.streak=(self.streak+sign if np.sign(self.streak)==sign else sign) if established else 0
            self.committed |= abs(self.streak)>=3
        self.tick+=1
        return results["semantic" if self.committed and self.tick<=24 else "cautious"][0]

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
        image=infra[None,None].astype(np.float32)/255
        score_result=self.loop.run_until_complete(self.student.infer(infra1=image))
        valid=(score_result["confidence"][0]>=.7)&np.isfinite(depth)&(depth>.15)&(depth<8)
        v,u=np.nonzero(valid)
        z=depth[v,u]
        optical=np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))
        xy=(optical@camera[:3,:3].T+camera[:3,3])[:,:2]-center
        forward=xy[:,0]*c+xy[:,1]*s
        lateral=-xy[:,0]*s+xy[:,1]*c
        i,j=np.floor((2.3-np.stack((forward,lateral)))/.2+.5).astype(int)
        inside=(i>=0)&(j>=0)&(i<24)&(j<24)
        scores=np.full(FWD.shape,np.inf,np.float32)
        np.minimum.at(scores,(i[inside],j[inside]),score_result["score"][0,v,u][inside])
        scores[~np.isfinite(scores)]=.5
        scores[blocked]=0
        self.semantic_projection_missing = not inside.any()
        if self.semantic_projection_missing:
            # Keep the normal stop on missing depth. This trial only substitutes
            # neutral scores when a depth frame with near returns is available.
            near_depth = np.isfinite(depth) & (depth > .15) & (depth < 2.3)
            if not self.allow_neutral_projection or np.count_nonzero(near_depth) < 64:
                raise ValueError("Infra1 projection did not observe any BEV cells")
        self.bev=np.stack((blocked,clearance,FWD/2.3,2*scores-1)).astype(np.float32)
        dx,dy=goal-center
        vector=np.asarray([(c*dx+s*dy)/4,(-s*dx+c*dy)/4,
                           velocity[0]/.55,velocity[1]/.9,*self.last_action],dtype=np.float32)
        self.replay_input = None
        if self.record_input:
            frame = np.zeros((97,24), dtype=np.float32)
            frame[:96] = self.bev.reshape(96,24)
            frame[96,:9] = (*vector, self.tick, self.streak, self.committed)
            self.replay_input = frame
        outputs=self.loop.run_until_complete(self.actor.infer(bev=self.bev[None],state=vector[None]))
        action=self.choose(outputs, neutral_projection=self.semantic_projection_missing)
        if not np.isfinite(action).all():
            raise ValueError("Semantic policy output is not finite")
        self.last_action=action
        speed=(float(action[0])+1)*.275
        return (speed if speed>=.05 else 0.,float(np.clip(action[1]*.9,-.75,.75)))

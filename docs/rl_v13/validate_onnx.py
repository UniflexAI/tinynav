import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tinynav.rl.policy import ActorCritic

torch.set_num_threads(4)
root = Path('tinynav_db/rl_runs/ppo_v13_carrot')
path = Path('tinynav/models/rl_nav_policy_v13.onnx')
ckpt = root / 'best.pt'
ck = torch.load(ckpt, map_location='cpu')
net = ActorCritic().eval()
net.load_state_dict(ck['model'])
graph = onnx.load(path)
onnx.checker.check_model(graph)
assert not any('critic' in x.name for x in graph.graph.initializer)
session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
assert [(x.name, x.shape) for x in session.get_inputs()] == [('bev', [1,3,64,64]), ('vec', [1,6]), ('h', [1,256])]
assert [(x.name, x.shape) for x in session.get_outputs()] == [('action', [1,2]), ('h_out', [1,256])]
rng = np.random.default_rng(26013)
max_action_error = max_hidden_error = 0.0
with torch.no_grad():
    for case in ('door_fixed', 'door_moving', 'wall_moving', 'clutter_moving'):
        trace = np.load(root / 'media' / f'{case}_v13.npz')
        h_pt, h_ort = net.initial_state(1, 'cpu'), np.zeros((1,256), np.float32)
        for tick in range(128):
            bev = trace['bev'][tick % len(trace['bev'])][None].astype(np.float32)
            vec = rng.uniform(-1,1,(1,6)).astype(np.float32)
            action_pt, h_pt = net.act(torch.from_numpy(bev), torch.from_numpy(vec), h_pt)
            action_ort, h_ort = session.run(None, dict(bev=bev, vec=vec, h=h_ort))
            max_action_error = max(max_action_error, float(np.max(np.abs(action_pt.numpy()-action_ort))))
            max_hidden_error = max(max_hidden_error, float(np.max(np.abs(h_pt.numpy()-h_ort))))
            assert np.isfinite(action_ort).all() and np.isfinite(h_ort).all()
            assert np.abs(action_ort).max() <= 1.0
assert max_action_error < 2e-5 and max_hidden_error < 2e-5
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = dict(model='v13', source_checkpoint=str(ckpt), checkpoint_step=ck['step'],
    checkpoint_sha256=sha(ckpt), selection_eval=ck['eval'], onnx=str(path),
    onnx_sha256=sha(path), onnx_bytes=path.stat().st_size, opset=17,
    deployment_parameters=sum(p.numel() for n,p in net.named_parameters() if not n.startswith('critic.') and n!='log_std'),
    validation=dict(checker='passed', provider='CPUExecutionProvider', recurrent_sequences=4,
        steps_per_sequence=128, seed=26013, inputs='recorded BEV + synthetic vectors in [-1,1]',
        max_action_abs_error=max_action_error, max_hidden_abs_error=max_hidden_error,
        torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__),
    training_config=ck['config'])
out = Path('docs/rl_v13/model_manifest.json')
out.write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))

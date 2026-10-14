"""Run reproducible initial planner comparisons against an isolated WebSim."""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8780')
parser.add_argument('--seconds', type=float, default=40)
parser.add_argument('--output', default='/tmp/continuous-websim-results.json')
parser.add_argument('--planner', choices=['both','continuous','dwa'], default='both')
parser.add_argument('--safety-radius', type=float, default=.08)
parser.add_argument('--dilation', type=int, default=1)
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()
def get(path):
    return json.load(urllib.request.urlopen(args.url + path, timeout=10))
def post(path, data):
    request = urllib.request.Request(args.url + path, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(request, timeout=10))
results = []
cases = [(p,s) for p in (['dwa','continuous'] if args.planner=='both' else [args.planner]) for s in ['l_turn','s_bend']]
for planner, scene in cases * args.repeat:
    config = get('/api/default-config')
    config['planner'] = planner
    config['robot']['safety_radius'] = args.safety_radius
    config['robot']['obstacle']['dilation_cells'] = args.dilation
    if scene == 's_bend':
        config['start'] = {'xy': [0., -.7], 'yaw_deg': 0.}
        config['target'] = [5.2, .7, 0.]
        boxes = [('lower_wall_entry',[1.25,-1.45,.65],[3.1,.22,1.3]),('upper_wall_entry',[1.15,.55,.65],[2.9,.22,1.3]),('lower_wall_exit',[4.05,-.35,.65],[3.1,.22,1.3]),('upper_wall_exit',[4.05,1.45,.65],[3.1,.22,1.3]),('left_deflector',[2.55,-.95,.65],[.22,.9,1.3]),('right_deflector',[3.25,.95,.65],[.22,.9,1.3])]
        config['objects'] = [dict(name=n,kind='box',center=c,size=s) for n,c,s in boxes]
    post('/api/update-config', {'config': config, 'reset': True})
    print('RUN', planner, scene, flush=True)
    began = time.monotonic()
    samples = []
    while time.monotonic() - began < args.seconds:
        time.sleep(.5)
        frame = get('/api/sim-state')['frame']
        distance = math.dist(frame['robot_xy'], config['target'][:2])
        samples.append(dict(t=time.monotonic()-began, xy=frame['robot_xy'], cmd=frame['selected_param'], collision=frame['collision'], distance=distance, planner_status=frame.get('planner_status', {})))
        if frame['collision'] or distance < .1:
            break
    result = dict(planner=planner, scene=scene, safety_radius=args.safety_radius, dilation=args.dilation, reached=distance < .1, final_distance=distance, collision=frame['collision'], elapsed=time.monotonic()-began, samples=samples)
    result['recovery_count'] = max((s['planner_status'].get('recovery_count', 0) for s in samples), default=0)
    result['blocked_samples'] = sum(s['planner_status'].get('reason') in ('rollout_collision','no_local_route','recovery_exhausted') for s in samples)
    results.append(result)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print({k:v for k,v in result.items() if k!='samples'}, flush=True)

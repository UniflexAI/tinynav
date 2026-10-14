import json,time,math,urllib.request
from pathlib import Path
b='http://127.0.0.1:8780'
def get(p):return json.load(urllib.request.urlopen(b+p,timeout=10))
def post(c,reset=False):
 return json.load(urllib.request.urlopen(urllib.request.Request(b+'/api/update-config',json.dumps({'config':c,'reset':reset}).encode(),{'Content-Type':'application/json'}),timeout=10))
def frame():return get('/api/sim-state')['frame']
results=[]
c=get('/api/default-config');c.update(planner='continuous',objects=[],target=[4.,0.,0.])
post(c,True);time.sleep(2.)
f=frame();assert f['selected_param'][0]>.09,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'forward','cmd':f['selected_param']})
c['paused']=True;post(c);time.sleep(.5)
f=frame();assert max(map(abs,f['selected_param']))<1e-6,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'pause','cmd':f['selected_param']})
c['paused']=False;post(c);time.sleep(1.)
f=frame();assert f['selected_param'][0]>.09,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'resume','cmd':f['selected_param']})
c['sensor_enabled']=False;post(c);time.sleep(1.2)
f=frame();assert max(map(abs,f['selected_param']))<1e-6,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
xy=f['robot_xy'];time.sleep(.4);f=frame();assert math.dist(xy,f['robot_xy'])<.001,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'sensor_loss_stop','cmd':f['selected_param']})
c['sensor_enabled']=True;post(c);time.sleep(1.)
f=frame();assert f['selected_param'][0]>.09,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'sensor_recovery','cmd':f['selected_param']})
c['nav_active']=False;post(c);time.sleep(.4)
f=frame();assert max(map(abs,f['selected_param']))<1e-6,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'nav_deactivation_stop','cmd':f['selected_param']})
c=get('/api/default-config');c['planner']='continuous'
c['guide_path']=[[i*.1,0.,0.] for i in range(41)]+[[4.,i*.1,0.] for i in range(1,45)]
c['guide_path'][-1]=c['target']
post(c,True);began=time.monotonic();samples=[]
while time.monotonic()-began<40:
 time.sleep(.5);f=frame();samples.append({'xy':f['robot_xy'],'cmd':f['selected_param'],'status':f['planner_status'],'collision':f['collision']})
 if f['collision'] or math.dist(f['robot_xy'],c['target'][:2])<.1:break
assert not f['collision'],{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
assert math.dist(f['robot_xy'],c['target'][:2])<.1,{k:f[k] for k in ('robot_xy','selected_param','planner_status','collision')}
results.append({'case':'map_guided_l_turn','elapsed':time.monotonic()-began,'distance':math.dist(f['robot_xy'],c['target'][:2]),'collision':f['collision'],'samples':samples})
Path('/tmp/pnc-fault-smoke-results.json').write_text(json.dumps(results,indent=2))
print([{k:v for k,v in r.items() if k!='samples'} for r in results],flush=True)

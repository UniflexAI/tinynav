"""Observe a local decision model during repeatable ROS web simulation runs."""
import json,time,statistics,argparse
from pathlib import Path
from urllib.request import Request,urlopen

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8766');p.add_argument('--output',required=True);p.add_argument('--timeout',type=float,default=30);p.add_argument('--repeats',type=int,default=2);a=p.parse_args()
 folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
 def api(path,body=None):
  req=Request(a.url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=35) as res:return json.load(res)
 catalog=api('/api/baseline/scenarios')['scenarios'];runs=[]
 api('/api/world-state/mode?mode=observed',{})
 try:
  for key in ('empty','open_target','l_turn','dead_end'):
   for repeat in range(a.repeats):
    scene=catalog[key];config=api('/api/default-config');config.update(name=scene['label'],scenario_id=key,start=scene['start'],target=scene['target'],objects=scene['objects']);config['camera']['max_range']=scene['cameraMaxRange']
    api('/api/baseline/start',{'config':config,'timeout_s':a.timeout});records=[];deadline=time.monotonic()+a.timeout+20;next_request=time.monotonic()+3
    while True:
     metrics=api('/api/baseline/status')
     if metrics['status']!='running':break
     if time.monotonic()>deadline:raise RuntimeError('Simulation deadline exceeded')
     if time.monotonic()>=next_request:
      submitted=api('/api/decision/evaluate',{});request_deadline=time.monotonic()+35
      while True:
       record=api('/api/decision/status')['record']
       if record['id']!=submitted['id']:raise RuntimeError('Observer request replaced by another caller')
       if record['status']!='running':break
       if time.monotonic()>request_deadline:raise RuntimeError('Observer deadline exceeded')
       time.sleep(.1)
      records.append(record);next_request=time.monotonic()+2
     time.sleep(.1)
    run={'scenario':key,'repeat':repeat+1,'baseline':api('/api/baseline/export'),'observations':records}
    runs.append(run);(folder/f'{key}-{repeat+1}.json').write_text(json.dumps(run,indent=2));print(key,repeat+1,metrics['status'],'goal',round(metrics['distance_to_goal_m'],2),'observations',len(records),flush=True)
 finally:api('/api/baseline/stop',{})
 complete=[r for run in runs for r in run['observations'] if r['status']=='complete'];lat=sorted(r['latency_ms'] for r in complete)
 summary={'mode':'observer_only','timeout_s':a.timeout,'runs':len(runs),'requests':sum(len(r['observations']) for r in runs),'completed':len(complete),'p50_ms':statistics.median(lat) if lat else None,'p95_ms':lat[max(0,int(.95*len(lat))-1)] if lat else None,'scenarios':{}}
 for key in catalog:
  selected=[r for r in runs if r['scenario']==key]
  if not selected:continue
  observations=[o for r in selected for o in r['observations'] if o['status']=='complete'];choices={}
  for o in observations:
   choice=o['response']['answers']['action']['choice'];choices[choice]=choices.get(choice,0)+1
  summary['scenarios'][key]={'outcomes':[r['baseline']['metrics']['status'] for r in selected],'final_goal_m':[round(r['baseline']['metrics']['distance_to_goal_m'],3) for r in selected],'choices':choices,'stuck_events':[r['baseline']['metrics']['stuck_events'] for r in selected],'mean_stuck_probability':statistics.mean(o['response']['answers']['stuck']['noul'] for o in observations) if observations else None}
 (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

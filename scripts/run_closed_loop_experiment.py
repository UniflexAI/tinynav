"""Run paired isolated ROS simulations; save outcomes and every policy event."""
import argparse,json,time,math,statistics
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor


def recovery(samples):
    first=None;reference=None;crossing=None
    for i,s in enumerate(samples):
        earlier=[p for p in samples[max(0,i-80):i+1] if s['t']-p['t']<=8]
        if not earlier:continue
        old=earlier[0]
        turned=abs((s['yaw_deg']-old['yaw_deg']+180)%360-180)
        if first is None and s['t']-old['t']>=3 and math.dist(s['xy'],old['xy'])<.1 and turned<15:
            first=s['t'];reference=s['distance_m']
        if first is not None:
            if s['collision'] or reference-s['distance_m']<.25:crossing=None
            elif crossing is None:crossing=s['t']
            elif s['t']-crossing>=5:return {'first_stuck_s':first,'recovered_s':crossing,'recovery_after_stuck_s':round(crossing-first,3)}
    return {'first_stuck_s':first,'recovered_s':None,'recovery_after_stuck_s':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=5);p.add_argument('--timeout',type=float,default=120);a=p.parse_args()
    if a.repeats<1 or not 5<=a.timeout<=120:p.error('repeats >=1; timeout 5..120')
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
    def worker(case,mode,port):
        base=f'http://127.0.0.1:{port}'
        def api(path,body=None):
            with urlopen(Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'}),timeout=35) as r:return json.load(r)
        ready_deadline=time.monotonic()+30
        while True:
            try:status=api('/api/experiment/status');break
            except (URLError,TimeoutError,OSError):
                if time.monotonic()>ready_deadline:raise
                time.sleep(.25)
        if status['mode']!=mode:raise RuntimeError('Wrong experiment mode')
        scene=api('/api/baseline/scenarios')['scenarios'][case];runs=[]
        try:
            for repeat in range(a.repeats):
                config=api('/api/default-config');config.update(scenario_id=case,name=scene['label'],start=scene['start'],target=scene['target'],objects=scene['objects']);config['camera']['max_range']=scene['cameraMaxRange']
                api('/api/baseline/start',{'config':config,'timeout_s':a.timeout});deadline=time.monotonic()+a.timeout+30
                while True:
                    metrics=api('/api/baseline/status')
                    if metrics['status']!='running':break
                    if time.monotonic()>deadline:raise RuntimeError('Simulator exceeded deadline')
                    time.sleep(1)
                baseline=api('/api/baseline/export');policy=api('/api/experiment/status')
                row={'scenario':case,'mode':mode,'repeat':repeat+1,'baseline':baseline,'policy':policy,'recovery':recovery(baseline['samples'])}
                (folder/f'{case}-{mode}-{repeat+1}.json').write_text(json.dumps(row,indent=2));runs.append(row)
                print(case,mode,repeat+1,metrics['status'],'goal',round(metrics['distance_to_goal_m'],3),'requests',sum(e['event']=='request' for e in policy['events']),'actions',sum(e['event']=='applied' for e in policy['events']),flush=True)
        finally:api('/api/baseline/stop',{})
        return runs
    jobs=[('l_turn','baseline',8770),('l_turn','model',8771),('dead_end','baseline',8772),('dead_end','model',8773)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,*j) for j in jobs];rows=[r for f in futures for r in f.result()]
    summary={'timeout_s':a.timeout,'repeats':a.repeats,'runs':len(rows),'groups':{},'recovery_definition':'After first motion-derived stuck episode, net goal progress >=0.25m retained for 5s without collision; separate from arrival.'}
    for case,mode,_ in jobs:
        chosen=[r for r in rows if r['scenario']==case and r['mode']==mode];m=[r['baseline']['metrics'] for r in chosen];events=[e for r in chosen for e in r['policy']['events']];times=[r['recovery']['recovery_after_stuck_s'] for r in chosen if r['recovery']['recovery_after_stuck_s'] is not None];fallbacks={}
        for e in events:
            if e['event'] not in ('request','applied','strategy_started','strategy_finished','stage_started'):fallbacks[e['event']]=fallbacks.get(e['event'],0)+1
        summary['groups'][case+':'+mode]={'runs':len(m),'arrived':sum(v['status']=='arrived' for v in m),'collision_runs':sum(v['collision_events']>0 for v in m),'timeouts':sum(v['status']=='timeout' for v in m),'final_goal_m':[round(v['distance_to_goal_m'],3) for v in m],'recovered':len(times),'recovery_after_stuck_s':times,'model_requests':sum(e['event']=='request' for e in events),'model_actions':sum(e['event']=='applied' for e in events),'strategy_outcomes':[e for e in events if e['event']=='strategy_finished'],'fallbacks':fallbacks,'latency_ms':[e['latency_ms'] for e in events if e.get('latency_ms') is not None]}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()

"""Evaluate frozen synthetic snapshots; never apply model commands to live sim."""
import argparse,json,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8766');p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=2);a=p.parse_args();folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
 def api(path,body=None):
  with urlopen(Request(a.url+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'}),timeout=35) as r:return json.load(r)
 catalog=api('/api/baseline/scenarios')['scenarios'];results=[]
 try:
  for case in ('open_target','l_turn','dead_end'):
   for repeat in range(a.repeats):
    scene=catalog[case];config=api('/api/default-config');config.update(scenario_id=case,start=scene['start'],target=scene['target'],objects=scene['objects']);config['camera']['max_range']=scene['cameraMaxRange']
    api('/api/baseline/start',{'config':config,'timeout_s':30});time.sleep(12)
    submitted=api('/api/decision/evaluate',{});api('/api/baseline/stop',{})
    for _ in range(160):
     record=api('/api/decision/status')['record']
     if record['id']!=submitted['id']:raise RuntimeError('Another caller replaced the request')
     if record['status']!='running':break
     time.sleep(.2)
    if record['status']!='complete':raise RuntimeError(str(record.get('error','request timeout')))
    result=api('/api/decision/compare?decision_id='+record['id'],{})
    result['repeat']=repeat+1
    (folder/f'{case}-{repeat+1}.json').write_text(json.dumps({'decision':record,'comparison':result},indent=2));results.append(result)
    print(case,repeat+1,result['choice'],'delta',result['progress_delta_m'],'collision',result['model']['collision'] if result['model'] else None,flush=True)
 finally:api('/api/baseline/stop',{})
 pairs=[r for r in results if r['model'] is not None]
 summary={'snapshots':len(results),'paired':len(pairs),'improved_over_0_05m':sum(r['progress_delta_m']>.05 and not r['model']['collision'] for r in pairs),'worse_over_0_05m':sum(r['progress_delta_m']<-.05 for r in pairs),'model_collisions':sum(r['model']['collision'] for r in pairs),'same_candidate':sum(r['baseline']['candidate_id']==r['model']['candidate_id'] for r in pairs),'kind':'3-second fixed-command branch, not closed-loop navigation','results':[{k:r[k] for k in ('scene','repeat','choice','confidence','progress_delta_m')} for r in results]}
 (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

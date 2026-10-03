"""Freeze actual measured snapshots; compare basic/v1/v2 without execution."""
import argparse,copy,json,time,statistics
from pathlib import Path
from urllib.request import Request,urlopen
from tool.simulator.decision_information import basic
from tool.simulator.decision_observer import planning_questions,validate_response
from tool.simulator.candidate_branches import compact_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--decisions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True);selected=[]
    for path in sorted(Path(a.runs).glob('*-model-*.json')):
        run=json.loads(path.read_text());sources=[]
        for event in run['policy']['events']:
            if event['event']!='request':continue
            path=Path(a.decisions)/(event['decision_id']+'.json')
            if not path.exists():continue
            source=json.loads(path.read_text())
            if source['status']=='complete' and source['context'].get('information_v1_state'):sources.append(source)
        if sources:
            selected.append((run['scenario'],run['repeat'],sources[0]))
            if len(sources)>1:selected.append((run['scenario'],run['repeat'],sources[-1]))
    rows=[];kinds=['basic','v1','v2']
    for i,(case,repeat,source) in enumerate(selected):
        v2=source['request']['state'];v1=source['context']['information_v1_state']
        planning=compact_report(source['context']['planning_report']);planning['snapshot_age_ms']=v1['planning']['snapshot_age_ms']
        states={'basic':basic(v1,planning),'v1':v1,'v2':v2};questions=planning_questions(v2)
        for state in states.values():
            if planning_questions(state)!=questions:raise ValueError('Questions changed')
            if [p['stages'] for p in state['recovery']['strategies']]!=[p['stages'] for p in v2['recovery']['strategies']]:raise ValueError('Commands changed')
        row={'scenario':case,'repeat':repeat,'source_id':source['id'],'evaluations':{},'source_audit':source['context']['observation_audit']}
        for kind in kinds[i%3:]+kinds[:i%3]:
            payload={'state':states[kind],'questions':questions};start=time.monotonic()
            with urlopen(Request('http://127.0.0.1:8090/v1/systemone',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=30) as response:result=json.load(response)
            validate_response(result,questions)
            row['evaluations'][kind]={'request':payload,'response':result,'latency_ms':round((time.monotonic()-start)*1000,1)}
        row['choices']={k:e['response']['answers']['alternative']['choice'] for k,e in row['evaluations'].items()};rows.append(row)
        (folder/('snapshot-'+str(i+1)+'.json')).write_text(json.dumps(row,indent=2));print(case,repeat,row['choices'],flush=True)
    if not rows:raise ValueError('No complete topology snapshots')
    summary={'snapshots':len(rows),'kind':'same_measured_snapshot_three_information_versions_no_execution','unchanged_questions_and_commands':True,
             'v1_to_v2_changed':sum(r['choices']['v1']!=r['choices']['v2'] for r in rows),'basic_to_v2_changed':sum(r['choices']['basic']!=r['choices']['v2'] for r in rows),'groups':{}}
    for case in ('l_turn','dead_end'):
        chosen=[r for r in rows if r['scenario']==case]
        summary['groups'][case]={k:{'choices':[r['choices'][k] for r in chosen],
                                   'median_latency_ms':statistics.median(r['evaluations'][k]['latency_ms'] for r in chosen)} for k in kinds}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()

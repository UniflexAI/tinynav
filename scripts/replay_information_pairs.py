"""Same-snapshot information ablation; no command execution."""
import argparse,copy,json,time,statistics
from pathlib import Path
from urllib.request import Request,urlopen
from tool.simulator.decision_information import basic
from tool.simulator.candidate_branches import compact_report
from tool.simulator.decision_observer import planning_questions,validate_response


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--decisions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True);selected=[]
    for path in sorted(Path(a.runs).glob('*-model-*.json')):
        row=json.loads(path.read_text());sources=[]
        for event in row['policy']['events']:
            if event['event']!='request':continue
            source=Path(a.decisions)/(event['decision_id']+'.json')
            if not source.exists():continue
            record=json.loads(source.read_text())
            if record['status']=='complete' and 'spatial_context' in record['request']['state']:sources.append(record)
        if sources:
            selected.append((row['scenario'],row['repeat'],sources[0]))
            if len(sources)>1:selected.append((row['scenario'],row['repeat'],sources[-1]))
    pairs=[]
    for i,(case,repeat,source) in enumerate(selected):
        rich=copy.deepcopy(source['request']['state']);planning=compact_report(source['context']['planning_report']);planning['snapshot_age_ms']=rich['planning']['snapshot_age_ms']
        states={'basic':basic(rich,planning),'rich':rich};questions=planning_questions(rich)
        if questions!=planning_questions(states['basic']):raise ValueError('Questions changed')
        pair={'scenario':case,'repeat':repeat,'source_decision_id':source['id'],'evaluations':{}}
        for kind in (['basic','rich'] if i%2==0 else ['rich','basic']):
            payload={'state':states[kind],'questions':questions};start=time.monotonic()
            with urlopen(Request('http://127.0.0.1:8090/v1/systemone',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'}),timeout=30) as response:result=json.load(response)
            elapsed=round((time.monotonic()-start)*1000,1);validate_response(result,questions)
            pair['evaluations'][kind]={'request':payload,'response':result,'latency_ms':elapsed}
        pair['choices']={k:v['response']['answers']['alternative']['choice'] for k,v in pair['evaluations'].items()}
        pairs.append(pair);(folder/('pair-'+str(i+1)+'.json')).write_text(json.dumps(pair,indent=2))
        print(case,repeat,pair['choices'],flush=True)
    if not pairs:raise ValueError('No completed enhanced snapshots')
    summary={'snapshots':len(pairs),'kind':'same_snapshot_information_ablation_no_execution','questions_and_candidates_identical':True,
             'choice_changed':sum(p['choices']['basic']!=p['choices']['rich'] for p in pairs),'groups':{}}
    for case in ('l_turn','dead_end'):
        chosen=[p for p in pairs if p['scenario']==case]
        summary['groups'][case]={k:{'strategy_choices':sum(p['choices'][k].startswith('strategy_') for p in chosen),
                                     'choices':[p['choices'][k] for p in chosen],
                                     'median_latency_ms':statistics.median(p['evaluations'][k]['latency_ms'] for p in chosen)} for k in ('basic','rich')}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()

"""Read experiment records and bounded loopback-only progress probes."""
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor

JOBS=[('l_turn','baseline',8770),('l_turn','model',8771),('dead_end','baseline',8772),('dead_end','model',8773)]


def results(folder):
    records=[]
    for path in sorted(folder.glob('*-*-*.json')):
        try:record=json.loads(path.read_text())
        except (OSError,ValueError):continue
        metrics=record['baseline']['metrics'];events=record['policy']['events']
        records.append({'scenario':record['scenario'],'mode':record['mode'],'repeat':record['repeat'],
                        'status':metrics['status'],'goal_m':metrics['distance_to_goal_m'],'elapsed_s':metrics['elapsed_s'],
                        'collisions':metrics['collision_events'],'recovery':record['recovery'],
                        'requests':sum(e['event']=='request' for e in events),'actions':sum(e['event']=='applied' for e in events),
                        'strategies':[{'id':e['strategy_id'],'outcome':e['outcome']} for e in events if e['event']=='strategy_finished']})
    summary_path=folder/'summary.json'
    try:summary=json.loads(summary_path.read_text())
    except (OSError,ValueError):summary=None
    def probe(job):
        case,mode,port=job
        try:
            with urlopen(f'http://127.0.0.1:{port}/api/baseline/status',timeout=.5) as response:metrics=json.load(response)
            with urlopen(f'http://127.0.0.1:{port}/api/experiment/status',timeout=.5) as response:policy=json.load(response)
        except (URLError,TimeoutError,OSError,ValueError):return {'scenario':case,'mode':mode,'available':False}
        return {'scenario':case,'mode':mode,'available':True,'metrics':metrics,'actions':sum(e['event']=='applied' for e in policy['events']),
                'requests':sum(e['event']=='request' for e in policy['events']),'last_event':policy['events'][-1] if policy['events'] else None}
    with ThreadPoolExecutor(max_workers=4) as pool:live=list(pool.map(probe,JOBS))
    return {'folder':folder.name,'complete':summary is not None,'summary':summary,'records':records,'live':live}

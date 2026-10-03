"""Observed center-cell connectivity and route evidence, not a planner."""
import copy
import math
from tool.simulator.decision_information import footprint_cells, fractions, relative
from tool.simulator.recovery_strategies import poses


def components(cells):
    free={c for c,v in cells.items() if v=='clear'};labels={};sizes={}
    for seed in sorted(free):
        if seed in labels:continue
        label=len(sizes);pending=[seed];labels[seed]=label;count=0
        while pending:
            x,y=pending.pop();count+=1
            for n in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if n in free and n not in labels:labels[n]=label;pending.append(n)
        sizes[label]=count
    return labels,sizes


def topology(state, xy, yaw, target, robot, cells, resolution, samples, observed_at, now):
    result=copy.deepcopy(state)
    cells={c:v for c,v in cells.items() if math.dist([(c[0]+.5)*resolution,(c[1]+.5)*resolution],xy)<=6}
    labels,sizes=components(cells)
    key=lambda p:(math.floor(p[0]/resolution),math.floor(p[1]/resolution))
    root=labels.get(key(xy));goal=labels.get(key(target))
    trail=[s['xy'] for s in samples[::max(1,len(samples)//160)]]
    route_labels={labels[key(p)] for p in trail if key(p) in labels}
    entry=trail[0] if trail else xy
    swept=set()
    for p in trail:swept.update(footprint_cells(p,robot,resolution))
    age=lambda c: max(0,now-observed_at[c]) if c in observed_at else None
    def freshness(sweep):
        clear=[c for c in sweep if cells.get(c)=='clear']
        ages=[age(c) for c in clear if age(c) is not None]
        return round(sum(v<=10 for v in ages)/max(1,len(sweep)),2)
    context=result['spatial_context'];context.pop('local_grid',None);context.pop('trajectory_breadcrumbs',None);context.pop('breadcrumb_columns',None)
    context['version']='observed_connectivity_v2'
    context['route']={'entry_xy_m':relative(entry,xy,yaw),'travelled_m':round(sum(math.dist(a['xy'],b['xy']) for a,b in zip(samples,samples[1:])),2),
                      'past_route_footprint':fractions(cells,swept),'recent_clear_fraction':freshness(swept),
                      'current_center_component_m2':round(sizes.get(root,0)*resolution**2,2),
                      'goal_center_connection':'unknown' if root is None or goal is None else 'observed_connected' if root==goal else 'not_connected_in_observed_cells'}
    context['stage_columns']='stage,blocked_fraction,unknown_fraction,near_past_path_fraction,recent_clear_fraction'
    context['endpoint_columns']='toward_entry_m,center_connection_to_past_route,component_m2'
    context['limits']=['Center-cell connectivity is not footprint clearance. Past route is historical. Recent means observed <=10s ago. Unknown is not clear; missing age is not recent.']
    for plan in result.get('recovery',{}).get('strategies',[]):
        p=list(xy);heading=yaw;rows=[]
        for stage,evidence in zip(plan['stages'],plan['observation_evidence']['stages']):
            sampled=list(poses(p,heading,[stage]));sweep=set()
            for position,_ in sampled:sweep.update(footprint_cells(position,robot,resolution))
            rows.append(evidence+','+str(freshness(sweep)));p,heading=sampled[-1]
        end=labels.get(key(p))
        connection='unknown' if end is None or not route_labels else 'connected' if end in route_labels else 'not_connected_in_observed_cells'
        plan['observation_evidence']['stages']=rows
        plan['observation_evidence']['endpoint']=','.join(map(str,[round(math.dist(xy,entry)-math.dist(p,entry),2),connection,round(sizes.get(end,0)*resolution**2,2)]))
    return result


def capture_audit(cells, observed_at, xy, resolution, now):
    return {'cells':[[int(i),int(j),v,float(observed_at[(i,j)]) if (i,j) in observed_at else None]
                     for (i,j),v in cells.items() if math.dist([(i+.5)*resolution,(j+.5)*resolution],xy)<=6],
            'resolution':float(resolution),'captured_monotonic':float(now)}

"""Bounded model evidence from measured cells and odometry, never scene boxes."""
import copy
import math
from tool.simulator.recovery_strategies import poses


def relative(point, origin, yaw):
    x,y=point[0]-origin[0],point[1]-origin[1]
    a=math.radians(yaw)
    return [round(math.cos(a)*x+math.sin(a)*y,2),round(-math.sin(a)*x+math.cos(a)*y,2)]


def footprint_cells(xy, robot, resolution):
    radius=math.hypot(robot.get('length',.4)/2+abs(robot.get('control_x',0)),robot.get('width',.3)/2)
    if robot.get('shape')=='circle':radius=max(radius,robot['radius'])
    n=math.ceil(radius/resolution)
    return {(math.floor((xy[0]+i*resolution)/resolution),math.floor((xy[1]+j*resolution)/resolution))
            for i in range(-n,n+1) for j in range(-n,n+1) if math.hypot(i*resolution,j*resolution)<=radius+resolution/2}


def fractions(cells, swept):
    n=max(1,len(swept))
    return {'clear':round(sum(cells.get(c)=='clear' for c in swept)/n,2),
            'blocked':round(sum(cells.get(c)=='blocked' for c in swept)/n,2),
            'unknown':round(sum(c not in cells for c in swept)/n,2)}


def enrich(state, xy, yaw, target, robot, cells, resolution, samples):
    result=copy.deepcopy(state)
    breadcrumbs=[]
    for sample in samples:
        if not breadcrumbs or math.dist(sample['xy'],breadcrumbs[-1]['xy'])>=.25 or abs((sample['yaw_deg']-breadcrumbs[-1]['yaw_deg']+180)%360-180)>=25:
            breadcrumbs.append(sample)
    if samples and (not breadcrumbs or samples[-1] is not breadcrumbs[-1]):breadcrumbs.append(samples[-1])
    # Retain coverage of the whole route rather than only the recent stall.
    if len(breadcrumbs)>16:
        breadcrumbs=[breadcrumbs[round(i*(len(breadcrumbs)-1)/15)] for i in range(16)]
    trail=[{'xy_m':relative(p['xy'],xy,yaw),'heading_deg':round((p['yaw_deg']-yaw+180)%360-180,1),
            'age_s':round(samples[-1]['t']-p['t'],1)} for p in breadcrumbs] if samples else []
    rows=[];a=math.radians(yaw)
    for forward in range(7,-8,-1):
        row=''
        for left in range(7,-8,-1):
            center=[xy[0]+math.cos(a)*forward*.4-math.sin(a)*left*.4,
                    xy[1]+math.sin(a)*forward*.4+math.cos(a)*left*.4]
            sector={(math.floor((center[0]+i*.1)/resolution),math.floor((center[1]+j*.1)/resolution)) for i in (-1,0,1) for j in (-1,0,1)}
            evidence=fractions(cells,sector)
            row+= 'R' if forward==left==0 else 'X' if evidence['blocked'] else '.' if evidence['clear']>=.7 else '?'
        rows.append(row)
    attempts=result.get('recovery',{}).get('previous_attempts',[])
    result['spatial_context']={'version':'observed_route_v1','frame':'current_robot_relative','axes':'x=forward, y=left, positive heading=left turn',
        'goal_xy_m':relative(target,xy,yaw),'trajectory_breadcrumbs':trail,
        'local_grid':{'rows':rows,'cell_m':.4,'legend':'Top=ahead, left=robot left; .=mostly measured clear, X=measured blocked, ?=unknown or partial, R=current center. Grid is not footprint clearance.'},
        'attempts_relative':[{'id':m['strategy_id'],'start_xy_m':relative(m['start_xy'],xy,yaw),
                              'outcome':m['outcome'],'post_planner_goal_progress_m':m.get('post_planner_goal_progress_m')} for m in attempts[-8:]],
        'limits':['Visited path shows past traversal, not current clearance. Retained cells have no per-cell age.',
                  'Unknown is never reclassified clear from path history. Stage evidence uses a conservative circular footprint.',
                  'Negative immediate goal progress can occur during retreat; subsequent escape or arrival is unproven.']}
    path_points=[p['xy'] for p in samples[::max(1,len(samples)//160)]]
    for plan in result.get('recovery',{}).get('strategies',[]):
        p=list(xy);heading=yaw;evidence=[]
        for stage in plan['stages']:
            sampled=list(poses(p,heading,[stage]));swept=set()
            for position,_ in sampled:swept.update(footprint_cells(position,robot,resolution))
            overlap=sum(any(math.dist(position,old)<=.25 for old in path_points) for position,_ in sampled)/max(1,len(sampled))
            p,heading=sampled[-1]
            evidence.append({'stage':stage['name'],'measured_footprint':fractions(cells,swept),
                             'near_traversed_path_fraction':round(overlap,2),'end_xy_m':relative(p,xy,yaw)})
        plan['observation_evidence']={'stages':evidence,'end_heading_relative_deg':round((heading-yaw+180)%360-180,1)}
    return compact_evidence(result)


def basic(state, original_planning=None):
    result=copy.deepcopy(state);result.pop('spatial_context',None)
    if original_planning is not None:result['planning']=copy.deepcopy(original_planning)
    note=result.get('recovery',{}).pop('shared_strategy_note',None)
    if note is not None:
        for plan in result['recovery']['strategies']:plan['observation_note']=note
    for plan in result.get('recovery',{}).get('strategies',[]):plan.pop('observation_evidence',None)
    return result


def compact_evidence(state):
    result=copy.deepcopy(state);context=result['spatial_context']
    planning=result.get('planning',{})
    if planning:
        planning['top_candidates']=[c for c in planning['top_candidates'] if c['id']==planning['selected_id']]
        planning.pop('collision_examples',None)
    plans=result.get('recovery',{}).get('strategies',[])
    notes={p.get('observation_note') for p in plans}
    if len(notes)==1 and None not in notes:
        result['recovery']['shared_strategy_note']=notes.pop()
        for plan in plans:plan.pop('observation_note')
    trail=context['trajectory_breadcrumbs']
    if len(trail)>4:trail=[trail[round(i*(len(trail)-1)/3)] for i in range(4)]
    context['trajectory_breadcrumbs']=[','.join(map(str,p['xy_m']+[p['heading_deg'],p['age_s']])) for p in trail]
    grid=context['local_grid'];grid['rows']=[row[3:12] for row in grid['rows'][3:12]]
    grid['legend']='Top=ahead, left=left; .=mostly measured clear, X=blocked, ?=unknown, R=robot. Not footprint clearance.'
    context['axes']='x forward, y left; heading positive left'
    context['breadcrumb_columns']='x_m,y_m,heading_deg,age_s'
    context['stage_columns']='stage,blocked_fraction,unknown_fraction,near_past_path_fraction'
    context['limits']=['Past traversal is not current clearance; cells have no age. Unknown stays unknown. Retreat may increase goal distance.']
    context.pop('attempts_relative',None)
    for plan in result.get('recovery',{}).get('strategies',[]):
        e=plan['observation_evidence'];stages=e['stages']
        e.pop('end_heading_relative_deg',None)
        e['stages']=[','.join(map(str,[s['stage'],s['measured_footprint']['blocked'],s['measured_footprint']['unknown'],s['near_traversed_path_fraction']])) for s in stages]
    return result

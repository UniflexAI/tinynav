"""Bounded candidate summaries and isolated synthetic-box rollouts."""
import copy
import math
from tool.simulator.planning_scene import SimObject, robot_hits_objects


def compact_report(report):
    rows = report['candidates']
    groups = {
        'forward': lambda c: c['linear_mps'] > 0 and abs(c['angular_radps']) < 1e-6,
        'left': lambda c: c['linear_mps'] > 0 and c['angular_radps'] < -1e-6,
        'right': lambda c: c['linear_mps'] > 0 and c['angular_radps'] > 1e-6,
        'reverse': lambda c: c['linear_mps'] < 0,
        'turn_left': lambda c: c['linear_mps'] == 0 and c['angular_radps'] < -1e-6,
        'turn_right': lambda c: c['linear_mps'] == 0 and c['angular_radps'] > 1e-6,
    }
    ranked = sorted((c for c in rows if c['cost'] is not None), key=lambda c: c['cost'])
    selected = next((c for c in rows if c['id'] == report['selected_id']), None)
    chosen = {selected['id']: selected} if selected else {}
    representatives = {}
    for name, predicate in groups.items():
        members = [c for c in rows if predicate(c)]
        valid = [c for c in ranked if predicate(c)]
        best = valid[0] if valid else None
        representatives[name] = {'count': len(members), 'collision_rejected': sum(c['cost'] is None for c in members), 'candidate_id': best['id'] if best else None}
        if best:
            chosen[best['id']] = best
    def concise(c):
        return {'id': c['id'], 'linear_mps': round(c['linear_mps'], 4), 'angular_radps': round(c['angular_radps'], 4),
                'control_linear_mps': round(c.get('control_linear_mps', c['linear_mps']), 4), 'control_yaw_radps': round(c.get('control_yaw_radps', -c['angular_radps']), 4), 'cost': round(c['cost'], 3) if c['cost'] is not None else None,
                'endpoint_world_xy': [round(x, 3) for x in c['endpoint_world_xy']],
                'obstacle_score': round(c['obstacle_score'], 3) if c['obstacle_score'] is not None else None,
                'reasons': c['reasons']}
    result = {k: v for k, v in report.items() if k not in ('candidates', 'selected_path_world_xy')}
    result['representatives'] = representatives
    result['top_candidates'] = [concise(c) for c in chosen.values()]
    result['collision_examples'] = [concise(c) for c in rows if c['cost'] is None][:2]
    result['notes'] = report['notes'] + ['angular_radps is camera-down-axis rotation; control yaw has the opposite sign.', 'Reverse gate is a planner penalty; a finite gated candidate is offered for isolated evaluation only.']
    return result


def rollout(config, seed, candidate, horizon=3.0, dt=0.05):
    if config.get('map_path'):
        raise ValueError('Branch evaluation supports synthetic boxes only')
    if candidate['cost'] is None:
        raise ValueError('Collision-rejected candidate cannot be evaluated as a model alternative')
    xy = list(seed['xy']); yaw = float(seed['yaw_deg'])
    robot = config['robot']; objects = [SimObject(**o) for o in config.get('objects', [])]
    start_distance = math.dist(xy, config['target'][:2]); path = [xy.copy()]
    collision = robot_hits_objects(xy, yaw, robot, objects)
    v = max(-robot['max_linear_vel'], min(robot['max_linear_vel'], candidate.get('control_linear_mps', candidate['linear_mps'])))
    w = max(-robot['max_angular_vel'], min(robot['max_angular_vel'], candidate.get('control_yaw_radps', -candidate['angular_radps'])))
    steps = round(horizon/dt)
    elapsed = 0.0
    for _ in range(steps):
        if collision: break
        angle = math.radians(yaw)
        new_xy = [xy[0] + math.cos(angle)*v*dt, xy[1] + math.sin(angle)*v*dt]
        new_yaw = (yaw + math.degrees(w*dt) + 180) % 360 - 180
        elapsed += dt
        if robot_hits_objects(new_xy, new_yaw, robot, objects):
            collision = True
            break
        xy, yaw = new_xy, new_yaw; path.append(xy.copy())
    end_distance = math.dist(xy, config['target'][:2])
    return {'candidate_id':candidate['id'], 'horizon_s':horizon,'dt_s':dt,'elapsed_s':round(elapsed,3),'collision':bool(collision),
            'goal_progress_m':round(start_distance-end_distance,4),'final_goal_m':round(end_distance,4),
            'final_xy':xy,'final_yaw_deg':yaw,'path_xy':path,'planner_penalties':copy.deepcopy(candidate['reasons'])}


def compare(record):
    report = record['context']['planning_report']; config = record['context']['scene_config']
    seed = {'xy':report['robot_world_xy'], 'yaw_deg':report['robot_yaw_deg']}
    candidates = {c['id']:c for c in report['candidates']}
    baseline = candidates.get(report['selected_id'])
    if baseline is None:raise ValueError('No selected baseline candidate')
    choice = record['response']['answers']['alternative']['choice']
    offered = {c['id'] for c in record['request']['state']['planning']['top_candidates'] if c['cost'] is not None}
    chosen = baseline if choice=='keep_current' else candidates.get(int(choice.removeprefix('candidate_'))) if choice.startswith('candidate_') else None
    if chosen and chosen['id'] not in offered:raise ValueError('Model candidate was not offered')
    results = {str(i):rollout(config,seed,candidates[i]) for i in offered}
    base = results[str(baseline['id'])]; model = results.get(str(chosen['id'])) if chosen else None
    return {'decision_id':record['id'],'plan_id':report['id'],'scene':record['context']['scenario'],'seed':seed,
            'kind':'isolated_constant_command_3s','choice':choice,'confidence':record['response']['answers']['alternative']['confidence'],
            'baseline':base,'model':model,'progress_delta_m':round(model['goal_progress_m']-base['goal_progress_m'],4) if model else None,
            'branches':results,'notes':['Each branch starts from the same report pose and immutable scene.', 'Ground-truth geometry is used for evaluation only, never model input.', 'Control command matches the first segment used by simulator_control; no replanning within the 3s branch.']}

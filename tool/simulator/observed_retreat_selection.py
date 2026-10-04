"""Rule-assisted retreat selection after active sensor observation.

This does not infer a model preference or certify unobserved geometry.
"""
import math
from tool.simulator.recovery_strategies import poses
from tool.simulator.decision_information import footprint_cells


def shortlist(plans, xy, yaw, robot, cells, resolution):
    eligible = []
    excluded = {}
    for plan in plans:
        if not plan['id'].startswith('retreat_'):
            excluded[plan['id']] = 'not_a_retreat'
            continue
        swept = set()
        for point, _ in poses(xy,yaw,plan['stages']):
            swept.update(footprint_cells(point,robot,resolution))
        if any(cells.get(c) != 'clear' for c in swept):
            excluded[plan['id']] = 'unknown_or_blocked_footprint'
            continue
        retreat = sum(max(0,-s['linear_mps'])*s['duration_s'] for s in plan['stages'])
        if retreat <= 0:
            excluded[plan['id']] = 'no_reverse_displacement'
            continue
        eligible.append((retreat,plan))
    selected = []
    for side in ('left','right'):
        candidates = [(length,p) for length,p in eligible if p['id'].endswith('_'+side)]
        if candidates:
            selected.append(max(candidates,key=lambda row:(row[0],row[1]['id']))[1])
    return {'plans':selected,'excluded':excluded,'rule':'longest_observed_clear_retreat_per_side',
            'current_safety_unproven':True,'model_preference':None}


def choose(screening, previous_attempts=()):
    plans = screening['plans']
    if not plans:
        return None
    # Equal evidence uses a declared, auditable rule; never fabricate model certainty.
    counts = {p['id']:sum(a.get('strategy_id')==p['id'] for a in previous_attempts) for p in plans}
    return min(plans,key=lambda p:(counts[p['id']],p['id']))

"""Offline local evidence screening before model usefulness assessment."""
from tool.simulator.independent_candidate_evaluation import observation_grade, consistent_selection

ORDERS = ('normal', 'reverse', 'rotate')


def screen(strategies, limit=2):
    ids = [plan['id'] for plan in strategies]
    if not ids or len(ids) != len(set(ids)) or limit < 1:
        raise ValueError('Invalid candidate set or limit')
    grades = {plan['id']: observation_grade(plan) for plan in strategies}
    eligible = [identity for identity in ids if grades[identity] == 'observed_probe']
    return {'grades': grades, 'eligible': eligible,
            'excluded': {identity: grade for identity, grade in grades.items() if identity not in eligible},
            'within_limit': len(eligible) <= limit,
            'observation_only': True, 'current_safety_unproven': True}


def selection(strategies, utility_by_order, limit=2):
    screening = screen(strategies, limit)
    if not screening['within_limit']:
        return {'choice': 'uncertain', 'reason': 'candidate_limit_exceeded'}
    eligible = set(screening['eligible'])
    if set(utility_by_order) != set(ORDERS):
        return {'choice': 'uncertain', 'reason': 'incomplete_orders'}
    if any(set(answers) != eligible for answers in utility_by_order.values()):
        return {'choice': 'uncertain', 'reason': 'incomplete_or_unexpected_candidate_assessments'}
    # Non-eligible candidates are locally excluded, not assessed by the model.
    grades = {order: dict(screening['grades']) for order in ORDERS}
    utility = {order: {identity: answers[identity] if identity in eligible else 'uncertain'
                       for identity in screening['grades']} for order, answers in utility_by_order.items()}
    return consistent_selection(strategies, grades, utility)

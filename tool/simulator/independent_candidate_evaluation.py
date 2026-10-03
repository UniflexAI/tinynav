"""Offline grading of existing recovery evidence; never authorizes execution."""
import copy
import math

GRADES = {'blocked': 0, 'partial_observation': 1, 'observed_probe': 2, 'uncertain': None}
CRITERIA = {
    'blocked': 'At least one stage has a positive measured blocked_fraction or observed_blocked_cells is positive.',
    'partial_observation': 'No measured blocking, but at least one stage has a positive unknown_fraction; full-sequence observation is incomplete.',
    'observed_probe': 'Every stage has zero measured blocked_fraction and zero unknown_fraction. Historical footprint labels support investigating this bounded sequence, not current safety or successful escape.',
    'uncertain': 'Stage evidence is missing, malformed or insufficient to assign one of the preceding observation grades.'}
INSTRUCTIONS = (
    'Grade the observation evidence for the single existing strategy in recovery.strategies. '
    'Evaluate this candidate independently; do not choose a side or compare with absent alternatives. '
    'Use each named stage blocked_fraction and unknown_fraction; whole-sequence summaries and center rays do not replace them. '
    'A positive blocked fraction takes precedence over unknown. Unknown takes precedence over observed_probe. '
    'Historical clear labels and recent_clear_fraction do not certify current safety. '
    'This is an evidence grade only: goal progress, direction, prior path and connectivity do not turn unknown into clear, '
    'and an observed_probe grade does not establish escape usefulness. Choose uncertain for incomplete evidence.')

USEFULNESS = {
    'investigate': 'Measured stage evidence and motion context support trying this bounded probe to gain local progress or observations; it does not prove escape.',
    'unsuitable': 'Measured blockage, zero useful motion, or a documented repeated failure with no changed evidence makes this probe unsuitable.',
    'uncertain': 'Unknown stage footprint, stale or incomplete evidence, or missing usefulness evidence prevents supporting this probe.'}
UTILITY_INSTRUCTIONS = (
    'Assess the single recovery strategy independently for a stalled robot away from its goal. '
    'Do not choose a direction by comparing with absent alternatives. '
    'Unknown footprint cells are not clear. Historical clear labels, center connectivity and past traversal do not certify current safety. '
    'Consider named per-stage fractions, recent evidence, displacement, goal bearing and previous attempts. '
    'Temporary negative goal progress during retreat is not by itself failure. '
    'Investigate means a supported bounded local test, not a safe command or proven escape route. '
    'Do not invent an exit or future planner success. Prefer uncertain if evidence does not support the probe.')


def usefulness_question(order):
    if len(order) != len(USEFULNESS) or set(order) != set(USEFULNESS):
        raise ValueError('Usefulness order must be a complete permutation')
    return {'type': 'choice', 'instructions': UTILITY_INSTRUCTIONS,
            'criteria': {key: USEFULNESS[key] for key in order}}


def grade_question(order):
    if len(order) != len(CRITERIA) or set(order) != set(CRITERIA):
        raise ValueError('Option order must be a complete permutation')
    return {'type': 'choice', 'instructions': INSTRUCTIONS,
            'criteria': {key: CRITERIA[key] for key in order}}


def candidate_state(named_state, strategy_id):
    result = copy.deepcopy(named_state)
    matches = [plan for plan in result['recovery']['strategies'] if plan['id'] == strategy_id]
    if len(matches) != 1:
        raise ValueError('Candidate identity is missing or duplicated')
    result['recovery']['strategies'] = matches
    result['recovery']['assessment_scope'] = 'One existing candidate only; omitted alternatives are evaluated separately. No command execution.'
    return result


def observation_grade(strategy):
    """Numerical reference, not model confidence or a collision/success certificate."""
    stages = strategy.get('stages', [])
    evidence = strategy.get('observation_evidence', {}).get('stages', [])
    if not stages or len(stages) != len(evidence):
        return 'uncertain'
    blocked, unknown = False, False
    cells = strategy.get('observed_blocked_cells')
    if not isinstance(cells, (int, float)) or isinstance(cells, bool) or not math.isfinite(cells) or cells < 0:
        return 'uncertain'
    blocked = cells > 0
    for command, row in zip(stages, evidence):
        if not isinstance(row, dict) or row.get('stage') != command['name']:
            return 'uncertain'
        for field in ('blocked_fraction', 'unknown_fraction'):
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                return 'uncertain'
        blocked = blocked or row['blocked_fraction'] > 0
        unknown = unknown or row['unknown_fraction'] > 0
    return 'blocked' if blocked else 'partial_observation' if unknown else 'observed_probe'


def consistent_selection(strategies, grades_by_order, usefulness_by_order, expected_orders=('normal', 'reverse', 'rotate')):
    """Require complete categorical agreement, valid observed evidence and a unique best candidate."""
    ids = [plan['id'] for plan in strategies]
    if len(ids) != len(set(ids)) or not ids:
        return {'choice': 'uncertain', 'reason': 'invalid_candidate_set'}
    if set(grades_by_order) != set(expected_orders) or set(usefulness_by_order) != set(expected_orders):
        return {'choice': 'uncertain', 'reason': 'incomplete_orders'}
    for answers in grades_by_order.values():
        if set(answers) != set(ids) or any(label not in GRADES for label in answers.values()):
            return {'choice': 'uncertain', 'reason': 'incomplete_or_invalid_grades'}
    for answers in usefulness_by_order.values():
        if set(answers) != set(ids) or any(label not in USEFULNESS for label in answers.values()):
            return {'choice': 'uncertain', 'reason': 'incomplete_or_invalid_usefulness'}
    unstable = [identity for identity in ids if len({answers[identity] for answers in grades_by_order.values()}) != 1 or
                len({answers[identity] for answers in usefulness_by_order.values()}) != 1]
    if unstable:
        return {'choice': 'uncertain', 'reason': 'order_inconsistent', 'unstable_candidates': unstable}
    stable = next(iter(grades_by_order.values()))
    if any(label == 'uncertain' for label in stable.values()):
        return {'choice': 'uncertain', 'reason': 'unresolved_candidate'}
    reference = {plan['id']: observation_grade(plan) for plan in strategies}
    if any(stable[identity] != reference[identity] for identity in ids):
        return {'choice': 'uncertain', 'reason': 'observation_grade_mismatch'}
    best = max(GRADES[label] for label in stable.values())
    if best < 2:
        return {'choice': 'uncertain', 'reason': 'no_fully_observed_probe'}
    leaders = [identity for identity in ids if GRADES[stable[identity]] == best]
    if len(leaders) != 1:
        return {'choice': 'uncertain', 'reason': 'tied_observed_candidates', 'leaders': leaders}
    utility = next(iter(usefulness_by_order.values()))
    if utility[leaders[0]] != 'investigate':
        return {'choice': 'uncertain', 'reason': 'no_consistent_useful_probe'}
    return {'choice': 'strategy_' + leaders[0], 'reason': 'unique_consistent_observed_probe',
            'advice_only': True, 'escape_usefulness_unproven': True}

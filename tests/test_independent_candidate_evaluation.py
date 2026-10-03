import copy
import math
import pytest
from tool.simulator.independent_candidate_evaluation import (
    CRITERIA, USEFULNESS, grade_question, usefulness_question, candidate_state,
    observation_grade, consistent_selection)


def plan(identity, unknown=0, blocked=0):
    return {'id': identity, 'observed_blocked_cells': 0,
            'stages': [{'name': 'turn', 'yaw_radps': .6}, {'name': 'probe', 'linear_mps': .3}],
            'observation_evidence': {'stages': [
                {'stage': name, 'unknown_fraction': unknown, 'blocked_fraction': blocked} for name in ('turn', 'probe')]}}


def orders(answers):
    return {key: copy.deepcopy(answers) for key in ('normal', 'reverse', 'rotate')}


def test_focused_state_preserves_commands_and_all_shared_evidence():
    source = {'goal': {'distance_m': 3}, 'planning': {'selected_id': 7},
              'recovery': {'strategies': [plan('pivot_left'), plan('pivot_right')],
                           'previous_attempts': [{'strategy_id': 'pivot_left', 'outcome': 'blocked'}]}}
    frozen = copy.deepcopy(source)
    focused = candidate_state(source, 'pivot_right')
    assert source == frozen
    assert focused['recovery']['strategies'] == [source['recovery']['strategies'][1]]
    assert focused['goal'] == source['goal'] and focused['planning'] == source['planning']
    assert focused['recovery']['previous_attempts'] == source['recovery']['previous_attempts']
    with pytest.raises(ValueError):
        candidate_state(source, 'absent')
    source['recovery']['strategies'].append(plan('pivot_right'))
    with pytest.raises(ValueError):
        candidate_state(source, 'pivot_right')


def test_question_permutations_preserve_criteria_and_instruction():
    for factory, criteria in ((grade_question, CRITERIA), (usefulness_question, USEFULNESS)):
        normal = factory(list(criteria))
        reverse = factory(list(criteria)[::-1])
        assert normal == reverse
        assert list(normal['criteria']) == list(criteria)
        assert list(reverse['criteria']) == list(criteria)[::-1]
        with pytest.raises(ValueError):
            factory([next(iter(criteria))] * len(criteria))


def test_observation_grade_uses_worst_stage_and_never_promotes_unknown():
    complete = plan('pivot_left')
    assert observation_grade(complete) == 'observed_probe'
    partial = plan('pivot_left', unknown=.001)
    assert observation_grade(partial) == 'partial_observation'
    partial['observation_evidence']['stages'][1]['blocked_fraction'] = .01
    assert observation_grade(partial) == 'blocked'
    complete['observed_blocked_cells'] = 1
    assert observation_grade(complete) == 'blocked'
    partial['observation_evidence']['stages'] = []
    assert observation_grade(partial) == 'uncertain'


@pytest.mark.parametrize('value', [None, -1, 1.1, math.nan, math.inf, True, '0'])
def test_invalid_numeric_evidence_abstains(value):
    candidate = plan('pivot_left')
    candidate['observation_evidence']['stages'][0]['unknown_fraction'] = value
    assert observation_grade(candidate) == 'uncertain'


def test_selector_requires_complete_agreement_and_unique_observed_candidate():
    strategies = [plan('pivot_right'), plan('retreat_60_left', unknown=.25)]
    grades = orders({'pivot_right': 'observed_probe', 'retreat_60_left': 'partial_observation'})
    usefulness = orders({'pivot_right': 'investigate', 'retreat_60_left': 'uncertain'})
    selection = consistent_selection(strategies, grades, usefulness)
    assert selection['choice'] == 'strategy_pivot_right' and selection['advice_only']
    assert selection['escape_usefulness_unproven']
    grades['reverse']['pivot_right'] = 'partial_observation'
    assert consistent_selection(strategies, grades, usefulness)['reason'] == 'order_inconsistent'
    grades['reverse']['pivot_right'] = 'observed_probe'
    usefulness['rotate']['pivot_right'] = 'uncertain'
    assert consistent_selection(strategies, grades, usefulness)['reason'] == 'order_inconsistent'
    usefulness['rotate']['pivot_right'] = 'investigate'
    del grades['reverse']['pivot_right']
    assert consistent_selection(strategies, grades, usefulness)['choice'] == 'uncertain'


def test_selector_cannot_use_usefulness_to_break_an_observation_tie():
    candidates = [plan('pivot_left'), plan('pivot_right')]
    grades = orders({'pivot_left': 'observed_probe', 'pivot_right': 'observed_probe'})
    utility = orders({'pivot_left': 'investigate', 'pivot_right': 'uncertain'})
    assert consistent_selection(candidates, grades, utility)['reason'] == 'tied_observed_candidates'


def test_selector_rejects_model_promotion_of_unknown_and_partial_evaluations():
    candidates = [plan('pivot_left', unknown=.25)]
    grades = orders({'pivot_left': 'observed_probe'})
    utility = orders({'pivot_left': 'investigate'})
    assert consistent_selection(candidates, grades, utility)['reason'] == 'observation_grade_mismatch'
    grades = orders({'pivot_left': 'partial_observation'})
    assert consistent_selection(candidates, grades, utility)['reason'] == 'no_fully_observed_probe'
    del grades['rotate']
    assert consistent_selection(candidates, grades, utility)['reason'] == 'incomplete_orders'
    candidates.append(plan('pivot_left'))
    assert consistent_selection(candidates, grades, utility)['reason'] == 'invalid_candidate_set'


def test_selector_abstains_if_unique_observed_candidate_has_no_usefulness_support():
    candidates = [plan('pivot_left')]
    grades = orders({'pivot_left': 'observed_probe'})
    utility = orders({'pivot_left': 'uncertain'})
    assert consistent_selection(candidates, grades, utility)['reason'] == 'no_consistent_useful_probe'

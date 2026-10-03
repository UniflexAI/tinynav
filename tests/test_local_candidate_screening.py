import copy
import pytest
from tool.simulator.local_candidate_screening import screen, selection


def candidate(identity, unknown=0):
    return {'id': identity, 'observed_blocked_cells': 0,
            'stages': [{'name': 'probe', 'linear_mps': .3, 'duration_s': 2}],
            'observation_evidence': {'stages': [{'stage': 'probe', 'blocked_fraction': 0, 'unknown_fraction': unknown}]}}


def orders(values):
    return {key: dict(values) for key in ('normal', 'reverse', 'rotate')}


def test_screen_preserves_candidates_and_excludes_any_unknown():
    candidates = [candidate('pivot_left'), candidate('retreat_60_left', .001)]
    original = copy.deepcopy(candidates)
    result = screen(candidates)
    assert result['eligible'] == ['pivot_left']
    assert result['excluded'] == {'retreat_60_left': 'partial_observation'}
    assert candidates == original and result['current_safety_unproven']


def test_cap_never_silently_picks_first_candidates():
    candidates = [candidate(str(i)) for i in range(3)]
    assert not screen(candidates)['within_limit']
    assert selection(candidates, {})['reason'] == 'candidate_limit_exceeded'
    with pytest.raises(ValueError):
        screen([candidate('x'), candidate('x')])


def test_unique_consistent_probe_is_advice_only():
    candidates = [candidate('pivot_right'), candidate('retreat_60_left', .25)]
    result = selection(candidates, orders({'pivot_right': 'investigate'}))
    assert result['choice'] == 'strategy_pivot_right' and result['advice_only']
    assert result['escape_usefulness_unproven']


def test_order_change_missing_or_extra_assessment_abstains():
    candidates = [candidate('pivot_right'), candidate('retreat_60_left', .25)]
    values = orders({'pivot_right': 'investigate'})
    values['reverse']['pivot_right'] = 'uncertain'
    assert selection(candidates, values)['reason'] == 'order_inconsistent'
    values = orders({'pivot_right': 'investigate'})
    del values['rotate']
    assert selection(candidates, values)['reason'] == 'incomplete_orders'
    values = orders({'pivot_right': 'investigate', 'retreat_60_left': 'investigate'})
    assert selection(candidates, values)['reason'] == 'incomplete_or_unexpected_candidate_assessments'


def test_ties_and_empty_eligible_set_do_not_force_direction():
    assert selection([candidate('left'), candidate('right')],
                     orders({'left': 'investigate', 'right': 'uncertain'}))['reason'] == 'tied_observed_candidates'
    assert selection([candidate('x', 1)], orders({}))['reason'] == 'no_fully_observed_probe'
    malformed = candidate('x')
    malformed['observation_evidence']['stages'] = []
    assert selection([malformed], orders({}))['reason'] == 'unresolved_candidate'

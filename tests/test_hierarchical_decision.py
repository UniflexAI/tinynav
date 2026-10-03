import copy
import pytest
from tool.simulator.hierarchical_decision import family_of, family_questions, member_questions
from tool.simulator.decision_observer import planning_questions


def state():
    strategies = []
    for direction in ('left', 'right'):
        strategies.append({'id': 'pivot_' + direction, 'stages': [{'name': 'turn'}, {'name': 'probe'}]})
        for distance in (60, 120, 180):
            strategies.append({'id': 'retreat_' + str(distance) + '_' + direction,
                               'stages': [{'name': 'retreat'}, {'name': 'turn'}, {'name': 'probe'}]})
    return {'planning': {'top_candidates': [], 'selected_id': 0}, 'recovery': {'strategies': strategies}}


def test_partition_preserves_all_commands_and_source():
    source = state()
    frozen = copy.deepcopy(source)
    families = family_questions(source)
    assert set(families['alternative']['criteria']) == {
        'keep_current', 'request_new_candidates', 'uncertain', 'pivot', 'retreat'}
    members = set()
    original = planning_questions(source)
    for family, count in (('pivot', 2), ('retreat', 6)):
        questions = member_questions(source, family)
        options = questions['alternative']['criteria']
        strategies = {key for key in options if key.startswith('strategy_')}
        assert len(strategies) == count
        assert {'keep_current', 'request_new_candidates', 'uncertain'} <= set(options)
        members.update(strategies)
        for key in ('action', 'stuck', 'planning_issue'):
            assert questions[key] == original[key]
    assert members == {key for key in original['alternative']['criteria'] if key.startswith('strategy_')}
    assert source == frozen


def test_unknown_stages_and_empty_family_fail_closed():
    with pytest.raises(ValueError):
        family_of({'stages': [{'name': 'teleport'}]})
    with pytest.raises(ValueError):
        member_questions(state(), 'unknown')
    source = state()
    source['recovery']['strategies'] = [p for p in source['recovery']['strategies'] if family_of(p) == 'pivot']
    assert 'retreat' not in family_questions(source)['alternative']['criteria']
    with pytest.raises(ValueError):
        member_questions(source, 'retreat')

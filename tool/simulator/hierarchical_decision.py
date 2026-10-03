"""Offline question organization; never generates or executes commands."""
import copy
from tool.simulator.decision_observer import planning_questions

FAMILIES = ('pivot', 'retreat')


def family_of(strategy):
    names = [stage['name'] for stage in strategy['stages']]
    if names == ['turn', 'probe']:
        return 'pivot'
    if names == ['retreat', 'turn', 'probe']:
        return 'retreat'
    raise ValueError('Unsupported recovery stages: ' + str(names))


def family_questions(state):
    questions = planning_questions(state)
    original = questions['alternative']
    criteria = {key: value for key, value in original['criteria'].items()
                if not key.startswith('strategy_')}
    groups = {family: [] for family in FAMILIES}
    for strategy in state['recovery']['strategies']:
        groups[family_of(strategy)].append(strategy['id'])
    for family, members in groups.items():
        if members:
            criteria[family] = ('At least one existing sequence in this family has sufficient measured evidence to investigate: '
                                + ', '.join(members))
    questions['alternative'] = {'type': 'choice', 'instructions': original['instructions'] +
        ' Select a family only if at least one member is supported by measured observations. '
        'Family size is not evidence. This selects no command; a second question must select an existing member. '
        'Pivot means turn then probe; retreat means retreat, turn then probe. Preserve abstention when evidence is insufficient.',
        'criteria': criteria}
    return questions


def member_questions(state, family):
    if family not in FAMILIES:
        raise ValueError('Unknown family')
    questions = planning_questions(state)
    original = questions['alternative']
    allowed = {'strategy_' + strategy['id'] for strategy in state['recovery']['strategies']
               if family_of(strategy) == family}
    if not allowed:
        raise ValueError('Empty family')
    questions['alternative'] = copy.deepcopy(original)
    questions['alternative']['criteria'] = {key: value for key, value in original['criteria'].items()
                                            if not key.startswith('strategy_') or key in allowed}
    questions['alternative']['instructions'] += (
        ' Independently assess the existing members of the ' + family +
        ' family. The preceding family selection is not safety evidence. '
        'Choose uncertain, keep_current or request_new_candidates if no member is supported. '
        'Select direction and duration only through an existing strategy ID.')
    return questions

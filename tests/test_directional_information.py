import copy
import json
import pytest
from tool.simulator.directional_information import explicit, mirror, compact_state_text, swap_direction
from tool.simulator.decision_observer import planning_questions


def state():
    strategies = []
    for side, sign in (('left', 1), ('right', -1)):
        for prefix in ('pivot', 'retreat_60'):
            stages = ([{'name': 'retreat', 'linear_mps': -0.3, 'yaw_radps': 0, 'duration_s': 2}]
                      if prefix.startswith('retreat') else [])
            stages += [{'name': 'turn', 'linear_mps': 0, 'yaw_radps': sign * 0.6, 'duration_s': 1.7},
                       {'name': 'probe', 'linear_mps': 0.3, 'yaw_radps': 0, 'duration_s': 2}]
            strategies.append({'id': prefix + '_' + side, 'stages': stages, 'final_yaw_deg': sign * 60,
                               'unknown_fraction': 0.25, 'observed_blocked_cells': 0,
                               'observation_evidence': {'stages': [s['name'] + ',0.0,0.25,0.5,0.1' for s in stages],
                                                        'endpoint': '-0.2,unknown,0.0'}})
    return {'goal': {'bearing': 'ahead_left', 'bearing_deg': 30},
            'robot': {'recent': {'turned_deg': 14.6}},
            'directions': {'left': {'known_clear_distance_m': 0.4, 'ends_in': 'unknown'},
                           'right': {'known_clear_distance_m': 1.2, 'ends_in': 'blocked'}},
            'planning': {'selected_id': 7, 'top_candidates': [
                {'id': 7, 'angular_radps': -0.3, 'control_yaw_radps': 0.3, 'endpoint_world_xy': [2, 3], 'reasons': []}],
                'robot_world_xy': [1, 2], 'target_world_xy': [3, 4], 'robot_yaw_deg': 25,
                'representatives': {'left': {'candidate_id': 19}, 'right': {'candidate_id': 25}}},
            'recovery': {'strategies': strategies, 'previous_attempts': [
                {'strategy_id': 'pivot_left', 'start_xy': [1, 2], 'outcome': 'blocked', 'post_planner_goal_progress_m': -0.1}]},
            'spatial_context': {'axes': 'x forward, y left', 'goal_xy_m': [2, 1],
                'route': {'entry_xy_m': [-1, 0.5]},
                'stage_columns': 'stage,blocked_fraction,unknown_fraction,near_past_path_fraction,recent_clear_fraction',
                'endpoint_columns': 'toward_entry_m,center_connection_to_past_route,component_m2'}}


def test_named_fields_preserve_source_commands_and_unknown():
    source = state()
    frozen = copy.deepcopy(source)
    named = explicit(source)
    assert source == frozen
    assert planning_questions(source) == planning_questions(named)
    assert [p['stages'] for p in source['recovery']['strategies']] == [p['stages'] for p in named['recovery']['strategies']]
    row = named['recovery']['strategies'][0]['observation_evidence']['stages'][0]
    assert row == {'stage': 'turn', 'blocked_fraction': 0.0, 'unknown_fraction': 0.25,
                   'near_past_path_fraction': 0.5, 'recent_clear_fraction': 0.1}
    assert named['recovery']['strategies'][0]['observation_evidence']['endpoint']['center_connection_to_past_route'] == 'unknown'


def test_reflection_swaps_rays_commands_history_and_world_points():
    source = state()
    reflected = mirror(source)
    assert mirror(reflected) == source
    assert reflected['directions']['left'] == source['directions']['right']
    assert reflected['goal'] == {'bearing': 'ahead_right', 'bearing_deg': -30}
    assert reflected['robot']['recent']['turned_deg'] == 14.6
    assert reflected['planning']['robot_world_xy'] == [1, -2]
    assert reflected['planning']['top_candidates'][0]['angular_radps'] == 0.3
    assert reflected['planning']['top_candidates'][0]['control_yaw_radps'] == -0.3
    assert reflected['recovery']['previous_attempts'][0]['strategy_id'] == 'pivot_right'
    assert reflected['recovery']['previous_attempts'][0]['outcome'] == 'blocked'
    assert reflected['recovery']['strategies'][0]['stages'][0]['yaw_radps'] == 0.6
    assert mirror(explicit(source)) == explicit(mirror(source))


def test_asymmetric_candidate_subset_restores_canonical_order():
    source = state()
    source['recovery']['strategies'] = [p for p in source['recovery']['strategies'] if p['id'] != 'pivot_left']
    assert mirror(mirror(source)) == source
    assert [p['id'] for p in mirror(source)['recovery']['strategies']] == ['pivot_left', 'retreat_60_left', 'retreat_60_right']
    assert swap_direction('strategy_retreat_60_left') == 'strategy_retreat_60_right'
    assert swap_direction('uncertain') == 'uncertain'


def test_compact_serialization_retains_index_annotation_without_data_loss():
    source = state()
    source['array'] = [{'x': i} for i in range(8)]
    compact = json.loads(compact_state_text(source))
    assert compact['array'] == [{'_index': i, 'x': i} for i in range(8)]
    del compact['array']
    del source['array']
    assert compact == source


@pytest.mark.parametrize('corruption', ['columns', 'name', 'fraction', 'connection'])
def test_malformed_evidence_fails_closed(corruption):
    source = state()
    evidence = source['recovery']['strategies'][0]['observation_evidence']
    if corruption == 'columns':
        source['spatial_context']['stage_columns'] = 'unknown'
    elif corruption == 'name':
        evidence['stages'][0] = 'retreat,0,0,0,0'
    elif corruption == 'fraction':
        evidence['stages'][0] = 'turn,0,nan,0,0'
    else:
        evidence['endpoint'] = '0,clear,1'
    with pytest.raises(ValueError):
        explicit(source)


def test_reading_oracle_compares_fractions_and_handles_missing_counts():
    from scripts.probe_direction_readout import expected_reading
    source = state()
    reps = source['planning']['representatives']
    reps['left'].update(count=2, collision_rejected=1)
    reps['right'].update(count=4, collision_rejected=1)
    assert expected_reading(source) == 'right'
    assert expected_reading(mirror(source)) == 'left'
    reps['right']['collision_rejected'] = 2
    assert expected_reading(source) == 'same'
    reps['right']['count'] = 0
    assert expected_reading(source) == 'missing'

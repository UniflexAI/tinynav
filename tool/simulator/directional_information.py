"""Offline evidence presentation and reflected counterfactuals, no controls."""
import copy
import math

DIRECTIONS = {'left': 'right', 'right': 'left', 'ahead_left': 'ahead_right',
              'ahead_right': 'ahead_left', 'behind_left': 'behind_right',
              'behind_right': 'behind_left', 'turn_left': 'turn_right', 'turn_right': 'turn_left'}
SIGNED = {'bearing_deg', 'robot_yaw_deg', 'yaw_deg', 'start_yaw_deg', 'end_yaw_deg',
          'angular_radps', 'control_yaw_radps', 'yaw_radps', 'final_yaw_deg',
          'heading_deg', 'end_heading_relative_deg'}
POINTS = {'robot_world_xy', 'target_world_xy', 'endpoint_world_xy', 'goal_xy_m', 'entry_xy_m',
          'start_xy', 'end_xy', 'start_xy_m', 'end_xy_m', 'xy_m'}
STAGE_FIELDS = ('stage', 'blocked_fraction', 'unknown_fraction', 'near_past_path_fraction', 'recent_clear_fraction')
END_FIELDS = ('toward_entry_m', 'center_connection_to_past_route', 'component_m2')


def swap_direction(value):
    if value in DIRECTIONS:
        return DIRECTIONS[value]
    for prefix in ('strategy_', ''):
        for suffix, replacement in (('_left', '_right'), ('_right', '_left')):
            if value.startswith(prefix) and value.endswith(suffix):
                return value[:-len(suffix)] + replacement
    return value


def explicit(state):
    result = copy.deepcopy(state)
    context = result['spatial_context']
    if context['stage_columns'] != ','.join(STAGE_FIELDS) or context['endpoint_columns'] != ','.join(END_FIELDS):
        raise ValueError('Unknown evidence columns')
    for plan in result['recovery']['strategies']:
        evidence = plan['observation_evidence']
        rows = []
        if len(evidence['stages']) != len(plan['stages']):
            raise ValueError('Stage evidence count mismatch')
        for command, text in zip(plan['stages'], evidence['stages']):
            values = text.split(',')
            if len(values) != len(STAGE_FIELDS) or values[0] != command['name']:
                raise ValueError('Stage evidence mismatch')
            row = {'stage': values[0]}
            for field, value in zip(STAGE_FIELDS[1:], values[1:]):
                number = float(value)
                if not math.isfinite(number) or not 0 <= number <= 1:
                    raise ValueError('Invalid evidence fraction')
                row[field] = number
            rows.append(row)
        values = evidence['endpoint'].split(',')
        if len(values) != 3 or values[1] not in ('connected', 'unknown', 'not_connected_in_observed_cells'):
            raise ValueError('Invalid endpoint evidence')
        evidence['stages'] = rows
        evidence['endpoint'] = dict(zip(END_FIELDS, (float(values[0]), values[1], float(values[2]))))
        plan['turn_direction'] = 'left' if plan['final_yaw_deg'] > 0 else 'right' if plan['final_yaw_deg'] < 0 else 'none'
    context.pop('stage_columns')
    context.pop('endpoint_columns')
    context['axes'] = 'Robot frame: x forward, y left; positive yaw turns left. World coordinates are separate.'
    context['evidence_definitions'] = {
        'blocked_fraction': 'Fraction of sampled conservative footprint cells labeled blocked.',
        'unknown_fraction': 'Fraction of sampled footprint cells without an occupancy label; not clear.',
        'near_past_path_fraction': 'Fraction of sampled stage centers within 0.25m of measured past path; historical only.',
        'recent_clear_fraction': 'Fraction of sampled footprint cells labeled clear and observed within 10s.',
        'toward_entry_m': 'Decrease in distance to historical route entry, not a safe escape route.',
        'center_connection_to_past_route': '4-neighbor observed-clear center-cell connectivity; not footprint safety.',
        'turned_deg': 'Absolute accumulated recent rotation, does not indicate left or right.'}
    return result


def mirror(state):
    """Reflect world y=0 and robot-local y=0 together; retain scalar observations."""
    def reflect(value, key=''):
        if isinstance(value, dict):
            mapped = {swap_direction(k): reflect(v, k) for k, v in value.items()}
            # Preserve direction display order, preventing a changed option position from masquerading as geometry.
            return {k: mapped[k] for k in value} if set(mapped) == set(value) else mapped
        if isinstance(value, list):
            if key in POINTS:
                if len(value) != 2:
                    raise ValueError('Unknown point dimension')
                return [value[0], -value[1]]
            return [reflect(v, key) for v in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key in SIGNED:
            return -value
        if isinstance(value, str) and key in ('bearing', 'id', 'strategy_id', 'turn_direction'):
            return swap_direction(value)
        return value
    result = reflect(copy.deepcopy(state))
    canonical = [prefix + '_' + side for side in ('left', 'right')
                 for prefix in ('pivot', 'retreat_60', 'retreat_120', 'retreat_180')]
    order = {name: index for index, name in enumerate(canonical)}
    mirrored = result['recovery']['strategies']
    if any(plan['id'] not in order for plan in mirrored):
        raise ValueError('Unsupported strategy ID in mirror')
    mirrored.sort(key=lambda plan: order[plan['id']])
    return result


def compact_state_text(state):
    """Match the runtime's array annotation, then remove only JSON whitespace."""
    import json
    def annotate(value):
        if isinstance(value, list):
            if len(value) >= 8:
                return [({'_index': index, **annotate(item)} if isinstance(item, dict)
                         else {'_index': index, 'value': annotate(item)}) for index, item in enumerate(value)]
            return [annotate(item) for item in value]
        if isinstance(value, dict):
            return {key: annotate(item) for key, item in value.items()}
        return value
    return json.dumps(annotate(state), ensure_ascii=False, separators=(',', ':'))

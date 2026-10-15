"""Persist the planner options exposed by the operate page."""
import json
import math
import os
from dataclasses import asdict
from tinynav.core import robot_specs


def presets():
    result = {}
    for name, robot in robot_specs.ROBOT_PRESETS.items():
        result[name] = dict(robot_type=name, safety_radius=robot.safety_radius,
            dilation_cells=robot.obstacle.dilation_cells, decay_factor=0.99,
            max_linear_vel=robot.max_linear_vel, max_angular_vel=robot.max_angular_vel,
            robot_z_bottom=robot.obstacle.robot_z_bottom, robot_z_top=robot.obstacle.robot_z_top)
    return result


def validate(values):
    defaults = presets()
    if not isinstance(values, dict) or not isinstance(values.get('robot_type'), str) or values['robot_type'] not in defaults:
        raise ValueError('Unsupported robot type')
    result = defaults[values['robot_type']].copy()
    if set(values) - set(result):
        raise ValueError('Unknown planner setting')
    result.update(values)
    for key, value in result.items():
        if key != 'robot_type' and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f'Invalid value: {key}')
    if result['safety_radius'] < 0 or not 0 <= result['decay_factor'] <= 1:
        raise ValueError('Safety radius must be non-negative; decay must be between 0 and 1')
    if not isinstance(result['dilation_cells'], int) or result['dilation_cells'] < 0:
        raise ValueError('Dilation must be a non-negative integer')
    if min(result['max_linear_vel'], result['max_angular_vel']) <= 0:
        raise ValueError('Maximum speeds must be positive')
    if result['robot_z_bottom'] >= result['robot_z_top']:
        raise ValueError('Z bottom must be below Z top')
    return result


class PlannerSettings:
    def __init__(self, root):
        self.path = os.path.join(root, 'planner_settings.json')
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.values = validate(json.load(f))
        else:
            name = robot_specs.ROBOT_CONFIG.name
            self.values = presets()[name].copy()
            self.values['decay_factor'] = float(os.environ.get('TINYNAV_OCCUPANCY_DECAY_FACTOR', '0.99'))
            self.values = validate(self.values)

    def save(self, values):
        values = validate(values)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temp = self.path + '.tmp'
        with open(temp, 'w') as f:
            json.dump(values, f, indent=2)
        os.replace(temp, self.path)
        self.values = values

    def environment(self, env):
        values = self.values
        robot = asdict(robot_specs.ROBOT_PRESETS[values['robot_type']])
        for key in ('safety_radius', 'max_linear_vel', 'max_angular_vel'):
            robot[key] = values[key]
        robot['min_linear_vel'] = min(robot['min_linear_vel'], robot['max_linear_vel'])
        robot['min_angular_vel'] = min(robot['min_angular_vel'], robot['max_angular_vel'])
        for key in ('dilation_cells', 'robot_z_bottom', 'robot_z_top'):
            robot['obstacle'][key] = values[key]
        return {**env, 'ROBOT_TYPE': values['robot_type'],
            'TINYNAV_ROBOT_CONFIG': json.dumps(robot),
            'TINYNAV_OCCUPANCY_DECAY_FACTOR': str(values['decay_factor'])}

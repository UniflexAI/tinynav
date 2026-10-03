"""Observation-only isolated ROS websim replay; scan and stationary controls."""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from tool.simulator.ros_planning_web import RosPlanningSimNode
from tool.simulator.recovery_strategies import poses, proposals
from tool.simulator.decision_information import footprint_cells
from tool.simulator.observed_topology import capture_audit


def evidence(plans, xy, yaw, robot, lab):
    result = {}
    for plan in plans:
        position, heading = list(xy), yaw
        rows = []
        for stage in plan['stages']:
            sampled = list(poses(position, heading, [stage]))
            cells = set()
            for point, _ in sampled:
                cells.update(footprint_cells(point, robot, lab.resolution))
            n = len(cells)
            unknown = sum(c not in lab.cells for c in cells)
            blocked = sum(lab.cells.get(c) == 'blocked' for c in cells)
            rows.append({'stage': stage['name'], 'cells': n, 'unknown_cells': unknown,
                         'blocked_cells': blocked, 'unknown_fraction': unknown / max(1, n),
                         'blocked_fraction': blocked / max(1, n)})
            position, heading = sampled[-1]
        result[plan['id']] = {'stages': rows,
                             'all_stages_exactly_observed_clear': all(r['unknown_cells'] == r['blocked_cells'] == 0 for r in rows)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots', required=True)
    p.add_argument('--decisions', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.getenv('ROS_DOMAIN_ID') != '218' or os.getenv('ROS_LOCALHOST_ONLY') != '1':
        raise ValueError('Requires dedicated local-only ROS domain 218')
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = RosPlanningSimNode()
    rows = []
    try:
        for index, path in enumerate(sorted(Path(a.snapshots).glob('snapshot-*.json'))):
            source = json.loads(path.read_text())
            saved = json.loads((Path(a.decisions) / (source['source_id'] + '.json')).read_text())
            state = source['evaluations']['v2']['request']['state']
            xy, yaw = state['planning']['robot_world_xy'], state['planning']['robot_yaw_deg']
            config = copy.deepcopy(saved['context']['scene_config'])
            if config.get('map_path'):
                raise ValueError('Synthetic observed scene required')
            config['start'] = {'xy': list(xy), 'yaw_deg': yaw}
            audit = source['source_audit']
            plans = state['recovery']['strategies']
            branches = ['scan', 'hold'] if index % 2 == 0 else ['hold', 'scan']
            row = {'source_id': source['source_id'], 'scenario': source['scenario'],
                   'source_file': str(path), 'config': config, 'fixed_candidates': plans, 'branches': {}}
            for branch in branches:
                node.set_config(config, reset=True)
                node.lab.begin(config, xy, yaw, timeout=90)
                node.lab.cells = {(int(i), int(j)): value for i, j, value, _ in audit['cells']}
                now = time.monotonic()
                node.lab.cell_observed_at = {(int(i), int(j)): now - max(0, audit['captured_monotonic'] - stamp)
                                            for i, j, _, stamp in audit['cells'] if stamp is not None}
                if node.lab.resolution != audit['resolution']:
                    raise ValueError('Observation resolution mismatch')
                node.running = True
                node.last_update = time.monotonic()
                before = evidence(plans, xy, yaw, config['robot'], node.lab)
                cells_before = capture_audit(node.lab.cells, node.lab.cell_observed_at, xy, node.lab.resolution, time.monotonic())
                start = time.monotonic()
                frames = []
                depths = {}
                progress = 0.0
                direction = 1 if index % 2 == 0 else -1
                speed = min(.6, config['robot']['max_angular_vel'])
                if speed <= 0:
                    raise ValueError('Robot cannot rotate')
                duration = 2 * math.pi / speed + .5
                while True:
                    elapsed = time.monotonic() - start
                    if node.collision or elapsed > 45:
                        break
                    if branch == 'scan' and progress >= 359.5:
                        break
                    if branch == 'hold' and elapsed >= duration:
                        break
                    command = Twist()
                    if branch == 'scan':
                        command.angular.z = direction * min(speed, math.radians(max(0, 360-progress)) / .2)
                    node.last_cmd = command
                    previous_yaw = node.yaw_deg
                    time.sleep(.125)
                    node.tick()
                    progress += abs((node.yaw_deg - previous_yaw + 180) % 360 - 180)
                    frames.append({'elapsed_s': time.monotonic() - start, 'xy': list(node.control_xy),
                                   'yaw_deg': node.yaw_deg, 'rotation_deg': progress, 'collision': bool(node.collision)})
                    bucket = int(progress // 90) if branch == 'scan' else 0
                    if str(bucket) not in depths:
                        depths[str(bucket)] = node.last_depth.copy()
                node.last_cmd = Twist()
                node.running = False
                elapsed = time.monotonic() - start
                after = evidence(plans, xy, yaw, config['robot'], node.lab)
                offered = proposals(xy, yaw, config['target'], config['robot'], node.lab.cells,
                                    node.lab.resolution, saved['request']['state']['recovery']['previous_attempts'])
                record = {'before': before, 'after': after, 'cells_before': cells_before,
                          'cells_after': capture_audit(node.lab.cells, node.lab.cell_observed_at, xy, node.lab.resolution, time.monotonic()),
                          'actual_end_xy': list(node.control_xy), 'actual_end_yaw_deg': node.yaw_deg,
                          'anchor_xy': xy, 'anchor_yaw_deg': yaw,
                          'position_drift_m': math.dist(xy, node.control_xy),
                          'return_heading_error_deg': abs((node.yaw_deg-yaw+180) % 360-180),
                          'rotation_deg': progress, 'duration_s': elapsed,
                          'collision': bool(node.collision), 'frames': frames,
                          'fresh_proposal_ids': [plan['id'] for plan in offered],
                          'scan_complete': branch == 'scan' and progress >= 359.5 and not node.collision,
                          'sensor_only_observations': True}
                np.savez_compressed(folder / (path.stem + '-' + branch + '-depth.npz'), **depths)
                row['branches'][branch] = record
                print(source['scenario'], path.stem, branch, 'rotation', round(progress, 1), 'collision', node.collision,
                      'exact clear', [k for k, v in after.items() if v['all_stages_exactly_observed_clear']], flush=True)
            rows.append(row)
            (folder / (path.stem + '.json')).write_text(json.dumps(row, indent=2))
    finally:
        node.running = False
        node.last_cmd = Twist()
        node.destroy_node()
        rclpy.shutdown()
    summary = {'kind': 'sensor_only_360_rotation_vs_stationary_from_frozen_measured_states',
               'snapshots': len(rows), 'branches': len(rows) * 2, 'ros_domain': 218,
               'no_planner_or_model_commands': True, 'groups': {}}
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        summary['groups'][scenario] = {branch: {
            'collision_runs': sum(r['branches'][branch]['collision'] for r in group),
            'completed_scans': sum(r['branches'][branch]['scan_complete'] for r in group),
            'max_position_drift_m': max(r['branches'][branch]['position_drift_m'] for r in group),
            'results': [{'source_id': r['source_id'],
                         'clear_before': [k for k, v in r['branches'][branch]['before'].items() if v['all_stages_exactly_observed_clear']],
                         'clear_after': [k for k, v in r['branches'][branch]['after'].items() if v['all_stages_exactly_observed_clear']],
                         'fresh_proposal_ids': r['branches'][branch]['fresh_proposal_ids']} for r in group]}
            for branch in ('hold', 'scan')}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

"""Read recorded Paths without ROS playback and verify motion conversion.

Usage (ROS sourced): python -m tool.field_control_regression --bags /bags --out result.json
No ROS nodes, publishers, robot SDKs or actuators are constructed.
"""
import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from geometry_msgs.msg import Twist
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Path as PathMsg
from scipy.spatial.transform import Rotation

from tinynav.platforms.cmd_vel_control import CmdVelControlNode, PATH_TRANSLATION_EPS_M


def controller():
    return SimpleNamespace(
        _nav_active=True, stair_mode=True, _stair_blocked=False, pose=object(),
        T_robot_to_camera=np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                                    [1., 0., 0., 0.], [0., 0., 0., 1.]]),
        lookahead_steps=1, planner_dt=.1, path_pose_stride=10,
        max_forward_speed=1., max_angular_speed=.75, fixed_reverse_speed=.3,
        force_turn_heading_threshold=np.deg2rad(80), last_path_update_time=None,
        path_period_ema=.12, latest_cmd=Twist(), logger=Mock(),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            to_msg=lambda: SimpleNamespace(sec=0, nanosec=0))))


def position(p):
    return np.array([p.position.x, p.position.y, p.position.z])


def rotation(p):
    q = p.orientation
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


def run(bags, baseline=None):
    rows = []
    for bag in sorted(bags.iterdir()):
        if not bag.is_dir() or not list(bag.glob('*.db3')):
            continue
        node, old = controller(), controller()
        result = dict(bag=bag.name, paths=0, stationary_paths=0, false_reverse_before=0,
                      false_reverse_after=0, wrong_stationary_yaw_before=0,
                      wrong_stationary_yaw_after=0, genuine_reverse_paths=0, failures=[])
        for db in sorted(bag.glob('*.db3')):
            with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
                topic = conn.execute("select id from topics where name='/planning/trajectory_path'").fetchone()
                if topic is None:
                    continue
                for timestamp, blob in conn.execute('select timestamp,data from messages where topic_id=? order by timestamp', topic):
                    msg = deserialize_message(blob, PathMsg)
                    if len(msg.poses) < 2:
                        continue
                    a, b = [p.pose for p in msg.poses[:2]]
                    body = rotation(a) @ node.T_robot_to_camera[:3, :3]
                    delta = body.T @ (position(b) - position(a))
                    turn = Rotation.from_matrix(body.T @ rotation(b) @ node.T_robot_to_camera[:3, :3]).as_rotvec()[2]
                    stationary = np.linalg.norm(delta[:2]) <= PATH_TRANSLATION_EPS_M
                    CmdVelControlNode.path_callback(node, msg)
                    v, w = node.latest_cmd.linear.x, node.latest_cmd.angular.z
                    result['paths'] += 1
                    if baseline is not None:
                        baseline(old, msg)
                    failures = []
                    if not np.isfinite([v, w]).all():
                        failures.append('nonfinite command')
                    if stationary:
                        result['stationary_paths'] += 1
                        expected = float(np.clip(turn, -.75, .75))
                        result['false_reverse_after'] += int(v < 0)
                        result['wrong_stationary_yaw_after'] += int(abs(w - expected) > 1e-9)
                        if v != 0 or abs(w - expected) > 1e-9:
                            failures.append('stationary path motion changed')
                        if baseline is not None:
                            result['false_reverse_before'] += int(old.latest_cmd.linear.x < 0)
                            result['wrong_stationary_yaw_before'] += int(abs(old.latest_cmd.angular.z - expected) > 1e-9)
                    if delta[0] < -PATH_TRANSLATION_EPS_M:
                        result['genuine_reverse_paths'] += 1
                        if (v, w) != (-.3, 0.):
                            failures.append('explicit reverse lost')
                    if failures:
                        result['failures'].append(dict(timestamp=timestamp, errors=failures, command=[v, w]))
        assert result['paths'], f'No planning paths in {bag}'
        rows.append(result)
    assert rows, 'No recorded paths found'
    return dict(passed=all(not r['failures'] for r in rows), scope='Recorded path conversion only; not a closed-loop physical replay', bags=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bags', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, help='Optional explicitly supplied old cmd_vel_control.py')
    args = parser.parse_args()
    baseline = None
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location('baseline_cmd_vel_control', args.baseline_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline = module.CmdVelControlNode.path_callback
    result = run(args.bags, baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

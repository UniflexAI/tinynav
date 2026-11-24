"""Read-only sensor/mask and recorded-Path control regression, without ROS playback.

Run with ROS sourced, on an isolated ROS domain with no robot controllers.
No executor is spun, original publishers are replaced before replay, and no SDK is loaded.
Recorded poses/targets and manual intervention are fixed inputs, not a physical counterfactual.
"""
import argparse
import contextlib
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation

from tinynav.core import planning_node as planning
from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.core.stair_obstacles import measured_tread_mask
from tinynav.platforms import cmd_vel_control as control


def source_key(msg):
    return msg.header.stamp.sec, msg.header.stamp.nanosec


def transform(msg):
    p = msg.pose.pose
    q = p.orientation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def run(bag, baseline_control, out):
    clock = SimpleNamespace(monotonic=lambda: 0.)
    old_clock, new_clock = baseline_control.time, control.time
    baseline_control.time = control.time = clock
    nodes = [baseline_control.CmdVelControlNode(), control.CmdVelControlNode()]
    emitted = [[], []]
    for node, rows in zip(nodes, emitted):
        node.cmd_pub = SimpleNamespace(publish=rows.append)
        node.logger = Mock()
    cfg = dataclasses.replace(ROBOT_CONFIG.obstacle, dilation_cells=planning.STAIR.dilation_cells,
                              min_wall_span_m=planning.STAIR.min_wall_span_m)
    grid = SimpleNamespace(origin=np.array([-2.5, -2.5, -.75]), grid_shape=(100, 100, 30),
        resolution=.05, step=10, occupancy_grid=np.zeros((100, 100, 30)), in_stair_mode=True,
        obstacle_config=ROBOT_CONFIG.obstacle)
    K = None
    status = ''
    status_rx = -np.inf
    camera_height = ROBOT_CONFIG.camera_height
    epoch = None
    guard = None
    paired = [{}, {}]
    masks, commands = [], []
    protected = dict(samples=0, violations=0)
    start = None

    def depth_frame(depth_msg, pose_msg, now):
        if K is None or guard is None or guard['blocked']:
            return
        stamp = source_key(depth_msg)
        cutoff = guard.get('accept_after')
        if cutoff is not None and stamp[0] + stamp[1] * 1e-9 < cutoff:
            return
        T = transform(pose_msg)
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(depth_msg.height, depth_msg.width)
        stair = now - status_rx < control.STAIR_MODE_TIMEOUT_S
        grid.in_stair_mode = stair
        planning.PlanningNode.update_occupancy_grid(grid, depth, T, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        use_treads = stair and status.startswith('up ') and ROBOT_CONFIG.name in ('go2', 'go2w')
        tread = measured_tread_mask(depth, T, K, grid.occupancy_grid, grid.origin, .05, camera_height) if use_treads else None
        active_cfg = cfg if stair else ROBOT_CONFIG.obstacle
        before = planning.build_obstacle_map(grid.occupancy_grid, grid.origin, .05, T[2, 3], active_cfg)
        after = planning.build_obstacle_map(grid.occupancy_grid, grid.origin, .05, T[2, 3], active_cfg, tread)
        assert not np.any(after & ~before), 'Tread classification introduced obstacles'
        raw = planning.build_obstacle_map(grid.occupancy_grid, grid.origin, .05, T[2, 3],
                                          dataclasses.replace(active_cfg, dilation_cells=0))
        clipped = np.any(grid.occupancy_grid[:, :, -2:] > cfg.occ_threshold, axis=2) & raw
        assert np.all(after[before & clipped]), 'Clipped wall was removed'
        if not use_treads:
            assert np.array_equal(after, before), 'Non-ascent obstacle behavior changed'
        xy = grid.origin[:2] + (np.indices(before.shape).transpose(1, 2, 0) + .5) * .05
        forward = T[:2, 2].copy()
        forward /= max(np.linalg.norm(forward), 1e-9)
        delta = xy - (T[:3, 3] - T[:3, :3] @ ROBOT_CONFIG.cam_offset_3d)[:2]
        along, lateral = delta @ forward, delta @ np.array([-forward[1], forward[0]])
        front = (along > .2) & (along < 1.2) & (np.abs(lateral) < .25)
        masks.append([now - start, int(before.sum()), int(after.sum()), int((before & front).sum()),
                      int((after & front).sum()), int(np.any(tread)) if tread is not None else 0,
                      int((before & clipped).sum())])

    try:
        for db in sorted(bag.glob('*.db3')):
            with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
                if start is None:
                    start = conn.execute('select timestamp from messages order by timestamp limit 1').fetchone()[0] / 1e9
                wanted = {'/slam/depth', '/slam/odometry_visual', '/slam/odometry', '/planning/trajectory_path',
                    '/cmd_vel', '/stair/status', '/nav/active', '/nav/paused', '/nav/vio_safety',
                    '/camera/camera/infra1/camera_info'}
                topics = {i: (name, get_message(tp)) for i, name, tp in conn.execute('select id,name,type from topics') if name in wanted}
                ids = list(topics)
                query = 'select timestamp,topic_id,data from messages where topic_id in (' + ','.join('?' for _ in ids) + ') order by timestamp'
                for timestamp, topic, blob in conn.execute(query, ids):
                    now = timestamp / 1e9
                    clock.monotonic = lambda: now
                    name, tp = topics[topic]
                    msg = deserialize_message(blob, tp)
                    if name.endswith('camera_info'):
                        K = np.array(msg.k).reshape(3, 3)
                    elif name == '/stair/status':
                        status, status_rx = msg.data, now
                        value = next((p.split('=', 1)[1] for p in status.split() if p.startswith('cam_h=')), None)
                        if value is not None:
                            camera_height = float(value)
                        for node in nodes:
                            node._on_stair_status(msg)
                    elif name == '/nav/vio_safety':
                        guard = json.loads(msg.data)
                        current = (guard['session'], guard['epoch'])
                        if current != epoch or guard['blocked']:
                            grid.occupancy_grid.fill(0.)
                            paired = [{}, {}]
                        epoch = current
                        for node in nodes:
                            node.vio_safety_callback(msg)
                    elif name in ('/slam/depth', '/slam/odometry_visual'):
                        which = int(name.endswith('visual'))
                        key = source_key(msg)
                        paired[which][key] = msg
                        if key in paired[1 - which]:
                            depth_frame(paired[0].pop(key), paired[1].pop(key), now)
                        if len(paired[which]) > 50:
                            paired[which].pop(next(iter(paired[which])))
                    elif name == '/slam/odometry':
                        for node in nodes:
                            node.pose_callback(msg)
                    elif name in ('/nav/active', '/nav/paused'):
                        for node in nodes:
                            (node._on_nav_active if name.endswith('active') else node._on_paused)(msg)
                    elif name == '/planning/trajectory_path':
                        for node in nodes:
                            node.path_callback(msg)
                    elif name == '/cmd_vel':
                        for rows in emitted:
                            rows.clear()
                        for node in nodes:
                            node.cmd_timer_callback()
                        if not all(emitted):
                            continue
                        before, after = [rows[-1] for rows in emitted]
                        node = nodes[1]
                        blocked = node.vio_blocked or node._paused or (node.stair_mode and node._stair_blocked)
                        if blocked:
                            protected['samples'] += 1
                            protected['violations'] += int(abs(after.linear.x) + abs(after.angular.z) > 1e-9)
                        fresh = node.stair_mode and not blocked and node.last_path_update_time is not None \
                            and now - node.last_path_update_time <= max(node.path_stale_slow_s, node.path_period_ema * node.path_stale_slow_factor)
                        if fresh and after.linear.x == 0. and after.angular.z != 0.:
                            assert abs(after.angular.z) >= ROBOT_CONFIG.stair_min_turn_vel - 1e-9
                        commands.append([now - start, before.linear.x, before.angular.z, after.linear.x,
                                         after.angular.z, int(fresh), int(blocked)])
        assert masks and commands and not protected['violations']
        m, c = np.array(masks), np.array(commands)
        tiny_before = (c[:, 1] == 0.) & (np.abs(c[:, 2]) > 1e-6) & (np.abs(c[:, 2]) < .25 - 1e-6) & (c[:, 5] > 0)
        tiny_after = (c[:, 3] == 0.) & (np.abs(c[:, 4]) > 1e-6) & (np.abs(c[:, 4]) < .25 - 1e-6) & (c[:, 5] > 0)
        result = dict(bag=bag.name, passed=not tiny_after.any(), frames=len(m), command_ticks=len(c),
            fresh_low_turns_before=int(tiny_before.sum()), fresh_low_turns_after=int(tiny_after.sum()),
            masks_changed=int(np.sum(m[:, 1] != m[:, 2])), measured_support_frames=int(m[:, 5].sum()),
            obstacle_cell_samples_before=int(m[:, 1].sum()), obstacle_cell_samples_after=int(m[:, 2].sum()),
            front_obstacle_cell_samples_before=int(m[:, 3].sum()), front_obstacle_cell_samples_after=int(m[:, 4].sum()),
            clipped_wall_cell_samples_preserved=int(m[:, 6].sum()), safety_stops=protected)
        np.savez_compressed(out / (bag.name + '.npz'), masks=m, commands=c)
        return result
    finally:
        for node in nodes:
            node.destroy_node()
        baseline_control.time, control.time = old_clock, new_clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bags', type=Path, required=True)
    parser.add_argument('--date', required=True, help='Exact recording date, e.g. 2026_11_24')
    parser.add_argument('--baseline-control', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bags = sorted(p for p in args.bags.glob(f'*{args.date}*') if p.is_dir() and list(p.glob('*.db3')))
    assert bags, 'No matching bags'
    args.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('baseline_cmd_vel_control', args.baseline_control)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rclpy.init()
    results = []
    try:
        for bag in bags:
            print('START', bag.name, flush=True)
            result = run(bag, baseline, args.out)
            results.append(result)
            (args.out / 'bags.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in results), bags=results,
                scope='Cold-start sensor/mask replay and control replay from recorded Paths at recorded command receipt times; no ROS playback, executor or SDK. Manual interventions and sensor poses remain fixed. No semantic wall ground truth or physical success-rate claim.'), indent=2))
            print('DONE', json.dumps(result), flush=True)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

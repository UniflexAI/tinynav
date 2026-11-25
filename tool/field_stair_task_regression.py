"""Replay stair target generation, intent gates and obstacle evidence without ROS playback.

Use an isolated localhost ROS domain. No executor is spun and no robot SDK is loaded.
Physical poses, manual intervention and old Path arrival times remain recorded inputs.
"""
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.time import Time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tinynav.core import planning_node as planning, stair_node
from tinynav.core.robot_specs import ROBOT_CONFIG
from tinynav.core.stair_obstacles import measured_tread_mask
from tinynav.platforms import cmd_vel_control as control
from tool.field_stair_regression import source_key, transform


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(bag, baseline, old_treads, out):
    info = json.loads((bag / 'info.json').read_text())
    direction = info['stair_status_at_save'].split()[0]
    assert direction in ('up', 'down')
    clock = SimpleNamespace(monotonic=lambda: 0.)
    old_clocks = stair_node.time, control.time
    stair_node.time = control.time = clock
    args = SimpleNamespace(camera_height=info['saved_stair_settings_at_start']['camera_height'],
        direction=direction, floors=0, landings_per_floor=2, turn='right' if direction == 'up' else 'left',
        memory=None, rate=2.)
    n, c = stair_node.StairNode(args), control.CmdVelControlNode()
    targets, paths, emitted = [], [], []
    status = ['']
    def publish_status(msg):
        status[0] = msg.data
        c._on_stair_status(msg)
    n.target_pub = SimpleNamespace(publish=targets.append)
    n.path_pub = SimpleNamespace(publish=paths.append)
    n.status_pub = SimpleNamespace(publish=publish_status)
    n.poi_change_pub = Mock()
    c.cmd_pub = SimpleNamespace(publish=emitted.append)
    c.logger = Mock()
    for node in (n, c):
        node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=clock.monotonic()))
    grids = [SimpleNamespace(origin=np.array([-2.5, -2.5, -.75]), grid_shape=(100, 100, 30),
        resolution=.05, step=10, occupancy_grid=np.zeros((100, 100, 30)), in_stair_mode=True,
        _stair_up=direction == 'up', obstacle_config=ROBOT_CONFIG.obstacle) for _ in range(2)]
    cfg = dataclasses.replace(ROBOT_CONFIG.obstacle, dilation_cells=planning.STAIR.dilation_cells,
                              min_wall_span_m=planning.STAIR.min_wall_span_m)
    guard = None
    epoch = None
    paired = [{}, {}]
    rows, commands, target_rows = [], [], []
    counters = dict(turn_ticks=0, turn_translation_violations=0, rejected_moving_paths=0,
        guarded_ticks=0, guarded_violations=0, moving_targets=0, turn_targets=0, stop_ticks=0)
    start = None
    camera_height = args.camera_height

    def paired_frame(depth_msg, pose_msg, now):
        if n.K is None:
            return
        n.depth_callback(depth_msg, pose_msg)
        if guard is None or guard['blocked']:
            return
        stamp = stair_node.stamp_sec(depth_msg.header.stamp)
        if guard.get('accept_after') is not None and stamp < guard['accept_after']:
            return
        T = transform(pose_msg)
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(depth_msg.height, depth_msg.width)
        masks, raw = [], None
        for i, (g, module, model) in enumerate(zip(grids, (baseline, planning), (old_treads, measured_tread_mask))):
            module.PlanningNode.update_occupancy_grid(g, depth, T, n.K[0, 0], n.K[1, 1], n.K[0, 2], n.K[1, 2])
            tread = model(depth, T, n.K, g.occupancy_grid, g.origin, .05, camera_height) if direction == 'up' else None
            mask = module.build_obstacle_map(g.occupancy_grid, g.origin, .05, T[2, 3], cfg) if i == 0 \
                else planning.build_obstacle_map(g.occupancy_grid, g.origin, .05, T[2, 3], cfg, tread)
            if i == 0 and tread is not None:
                # The baseline function has the same optional tread-mask API.
                mask = module.build_obstacle_map(g.occupancy_grid, g.origin, .05, T[2, 3], cfg, tread)
            if i == 1:
                raw = planning.build_obstacle_map(g.occupancy_grid, g.origin, .05, T[2, 3],
                                                  dataclasses.replace(cfg, dilation_cells=0))
                clipped = np.any(g.occupancy_grid[:, :, -2:] > .1, axis=2) & raw
                assert np.all(mask[clipped]), 'Tread filter removed a clipped wall candidate'
            masks.append(mask)
        if direction == 'down':
            assert np.array_equal(grids[0].occupancy_grid, grids[1].occupancy_grid)
            assert np.array_equal(*masks), 'Downstairs obstacle mapping changed'
        g = grids[1]
        xy = g.origin[:2] + (np.indices(masks[0].shape).transpose(1, 2, 0) + .5) * .05
        forward = T[:2, 2].copy(); forward /= max(np.linalg.norm(forward), 1e-9)
        delta = xy - T[:2, 3]
        along, lateral = delta @ forward, delta @ np.array([-forward[1], forward[0]])
        front = (along > .2) & (along < 1.2) & (np.abs(lateral) < .3)
        rows.append([now - start, masks[0].sum(), masks[1].sum(), (masks[0] & front).sum(),
                     (masks[1] & front).sum(), clipped.sum()])
        if len(rows) % 500 == 0:
            print('FRAMES', bag.name, len(rows), flush=True)

    try:
        for db in sorted(bag.glob('*.db3')):
            with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as conn:
                if start is None:
                    start = conn.execute('select timestamp from messages order by timestamp limit 1').fetchone()[0] / 1e9
                wanted = {'/slam/depth', '/slam/odometry_visual', '/slam/odometry', '/planning/trajectory_path',
                    '/control/target_pose', '/cmd_vel', '/stair/status', '/nav/active', '/nav/paused',
                    '/nav/vio_safety', '/camera/camera/infra1/camera_info'}
                topics = {i: (name, get_message(tp)) for i, name, tp in conn.execute('select id,name,type from topics') if name in wanted}
                ids = list(topics)
                query = 'select timestamp,topic_id,data from messages where topic_id in (' + ','.join('?' for _ in ids) + ') order by timestamp'
                for timestamp, topic, blob in conn.execute(query, ids):
                    now = timestamp / 1e9
                    clock.monotonic = lambda: now
                    name, tp = topics[topic]
                    msg = deserialize_message(blob, tp)
                    if name.endswith('camera_info'):
                        n.info_callback(msg)
                    elif name == '/stair/status':
                        value = next((p.split('=', 1)[1] for p in msg.data.split() if p.startswith('cam_h=')), None)
                        if value is not None:
                            camera_height = float(value)
                    elif name == '/nav/vio_safety':
                        guard = json.loads(msg.data)
                        current = (guard['session'], guard['epoch'])
                        if current != epoch or guard['blocked']:
                            for g in grids:
                                g.occupancy_grid.fill(0.)
                            paired = [{}, {}]
                        epoch = current
                        n.vio_safety_callback(msg); c.vio_safety_callback(msg)
                    elif name in ('/slam/depth', '/slam/odometry_visual'):
                        which = int(name.endswith('visual'))
                        key = source_key(msg)
                        paired[which][key] = msg
                        if key in paired[1 - which]:
                            paired_frame(paired[0].pop(key), paired[1].pop(key), now)
                        if len(paired[which]) > 50:
                            paired[which].pop(next(iter(paired[which])))
                    elif name == '/slam/odometry':
                        n.odom_callback(msg); c.pose_callback(msg)
                    elif name == '/nav/active':
                        c._on_nav_active(msg)
                    elif name == '/nav/paused':
                        n.pause_callback(msg); c._on_paused(msg)
                    elif name == '/control/target_pose':
                        targets.clear(); paths.clear()
                        n.timer_callback()
                        if not targets:
                            counters['stop_ticks'] += 1
                            target_rows.append(dict(t=now - start, status=status[0], target=None))
                            continue
                        target = targets[-1].pose.pose.position
                        xyz = np.array([target.x, target.y, target.z])
                        turning = 'intent=turn' in status[0].split()
                        if turning:
                            counters['turn_targets'] += 1
                            q = Rotation.from_matrix(n.latest_T[:3, :3]).as_quat()
                            _, param = planning.generate_trajectories(n.latest_T[:3, 3], q,
                                min_forward_speed=.25, min_turn_speed=.25, turn_in_place=True)
                            assert np.all(param[:, 0] == 0.)
                        else:
                            counters['moving_targets'] += 1
                            assert paths and paths[-1].poses, 'Moving target has no route'
                            points = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in paths[-1].poses])
                            assert np.linalg.norm(points - xyz, axis=1).min() < 1e-7, 'Target detached from route'
                            if direction == 'down':
                                assert xyz[2] <= n.latest_T[2, 3] - n.gen.camera_height + n.gen.cfg.max_step + 1e-7
                        target_rows.append(dict(t=now - start, status=status[0], target=xyz.tolist(),
                            camera=n.latest_T[:3, 3].tolist(), camera_height=n.gen.camera_height,
                            turning=turning, path_poses=len(paths[-1].poses) if paths else 0))
                    elif name == '/planning/trajectory_path':
                        if c.stair_mode and getattr(c, '_stair_turning', False) and len(msg.poses) > 1:
                            p, q = msg.poses[0].pose.position, msg.poses[1].pose.position
                            counters['rejected_moving_paths'] += int(np.hypot(q.x - p.x, q.y - p.y) > 1e-3)
                        c.path_callback(msg)
                    elif name == '/cmd_vel':
                        emitted.clear(); c.cmd_timer_callback()
                        if not emitted:
                            continue
                        cmd = emitted[-1]
                        turning = c.stair_mode and getattr(c, '_stair_turning', False)
                        guarded = c.vio_blocked or c._paused or (c.stair_mode and c._stair_blocked)
                        counters['turn_ticks'] += int(turning)
                        counters['turn_translation_violations'] += int(turning and abs(cmd.linear.x) > 1e-9)
                        counters['guarded_ticks'] += int(guarded)
                        counters['guarded_violations'] += int(guarded and abs(cmd.linear.x) + abs(cmd.angular.z) > 1e-9)
                        commands.append([now - start, cmd.linear.x, cmd.angular.z, int(turning), int(guarded)])
        assert rows and target_rows and counters['moving_targets'] and counters['turn_targets']
        assert not counters['turn_translation_violations'] and not counters['guarded_violations']
        m = np.asarray(rows)
        result = dict(bag=bag.name, passed=True, recorded_commit=info['commit'], direction=direction,
            paired_frames=len(rows), targets=len(target_rows), commands=len(commands), counters=counters,
            obstacle_cell_samples_before=int(m[:, 1].sum()), obstacle_cell_samples_after=int(m[:, 2].sum()),
            front_cell_samples_before=int(m[:, 3].sum()), front_cell_samples_after=int(m[:, 4].sum()),
            clipped_wall_cell_samples_preserved=int(m[:, 5].sum()))
        (out / (bag.name + '-targets.json')).write_text(json.dumps(target_rows, indent=2))
        np.savez_compressed(out / (bag.name + '.npz'), masks=m, commands=commands)
        return result
    finally:
        n.destroy_node(); c.destroy_node()
        stair_node.time, control.time = old_clocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bags', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    baseline = load_module('stair_baseline_planning', args.baseline_root / 'tinynav/core/planning_node.py')
    old = load_module('stair_baseline_treads', args.baseline_root / 'tinynav/core/stair_obstacles.py')
    args.out.mkdir(parents=True, exist_ok=True)
    bags = sorted(p for p in args.bags.glob(f'*{args.date}*') if p.is_dir() and list(p.glob('*.db3')))
    assert bags
    results = []
    rclpy.init()
    try:
        for bag in bags:
            print('START', bag.name, flush=True)
            result = run(bag, baseline, old.measured_tread_mask, args.out)
            results.append(result)
            (args.out / 'bags.json').write_text(json.dumps(dict(passed=True, bags=results,
                scope='Cold-start production StairNode target/intent replay, old Path injection and source-paired obstacle evidence. Manual physical poses remain fixed; not a physical success-rate claim. No executor, SDK or ROS playback.'), indent=2))
            print('DONE', json.dumps(result), flush=True)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

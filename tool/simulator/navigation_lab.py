"""Baseline measurements and text observations; no ROS or policy dependency."""
from collections import deque
import copy
import hashlib
import json
import math
import time
import uuid
import numpy as np

DIRECTIONS = [('ahead', 0), ('ahead_left', 45), ('left', 90), ('behind_left', 135),
              ('behind', 180), ('behind_right', -135), ('right', -90), ('ahead_right', -45)]


def bearing_name(angle):
    return min(DIRECTIONS, key=lambda item: abs((angle-item[1]+180) % 360-180))[0]


class NavigationLab:
    resolution = 0.1

    def __init__(self):
        self.reset()

    def reset(self):
        self.cells = {}
        self.history = deque(maxlen=160)
        self.run = None
        self.samples = []

    def begin(self, config, xy, yaw, timeout=120):
        self.reset()
        self.run = {'id': uuid.uuid4().hex, 'scenario': config.get('scenario_id', config.get('name','custom')), 'config': copy.deepcopy(config), 'status': 'running',
                    'timeout_s': timeout, 'arrival_radius_m': 0.35, 'path_length_m': 0.0,
                    'collision_events': 0, 'stuck_events': 0, 'elapsed_s': 0.0,
                    'distance_to_goal_m': math.dist(xy, config['target'][:2]),
                    'started_at_unix': time.time(),
                    'config_hash': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]}
        self.started = time.monotonic()
        self.previous_xy = list(xy)
        self.was_collision = False
        self.was_stuck = False

    def observe_depth(self, depth, transform, camera, robot):
        # Only measured rays carve free space. Zero/no-return depth remains unknown.
        h, w = depth.shape
        v, u = np.mgrid[0:h:4, 0:w:4]
        z = depth[v, u].ravel()
        valid = np.isfinite(z) & (z > 0)
        u, v, z = u.ravel()[valid], v.ravel()[valid], z[valid]
        points = np.column_stack(((u-(w-1)/2)/camera['fx']*z,
                                  (v-(h-1)/2)/camera['fy']*z, z))
        endpoints = points @ transform[:3, :3].T + transform[:3, 3]
        origin = transform[:3, 3]
        ground = float(camera.get('ground_z', 0))
        band = robot.get('obstacle', {})
        bottom = max(ground+0.05, origin[2]+float(band.get('robot_z_bottom', -0.4)))
        top = origin[2]+float(band.get('robot_z_top', 0.4))
        free, occupied = set(), set()
        for endpoint in endpoints:
            length = np.linalg.norm(endpoint-origin)
            n = max(2, int(length/0.07))
            points = origin + np.arange(n)[:, None]/n*(endpoint-origin)
            points = points[(points[:, 2] >= bottom) & (points[:, 2] <= top)]
            free.update(map(tuple, np.floor(points[:, :2]/self.resolution).astype(int)))
            if bottom <= endpoint[2] <= top:
                occupied.add(tuple(np.floor(endpoint[:2]/self.resolution).astype(int)))
        for cell in free:
            if self.cells.get(cell) != 'blocked':
                self.cells[cell] = 'clear'
        for cell in occupied:
            self.cells[cell] = 'blocked'
        if len(self.cells) > 50000:
            center = origin[:2]/self.resolution
            self.cells = {k: val for k, val in self.cells.items() if math.dist(k, center) < 120}

    def update(self, xy, yaw, target, collision, now=None):
        now = time.monotonic() if now is None else now
        distance = math.dist(xy, target[:2])
        self.history.append((now, list(xy), yaw, distance))
        recent = self.recent(now)
        if self.run and self.run['status'] == 'running':
            r = self.run
            r['elapsed_s'] = round(now-self.started, 3)
            r['path_length_m'] += math.dist(xy, self.previous_xy)
            self.previous_xy = list(xy)
            r['distance_to_goal_m'] = distance
            if collision and not self.was_collision:
                r['collision_events'] += 1
            stuck = recent['pattern'] == 'stuck'
            if stuck and not self.was_stuck:
                r['stuck_events'] += 1
            self.was_stuck, self.was_collision = stuck, collision
            self.samples.append({'t': r['elapsed_s'], 'xy': list(xy), 'yaw_deg': yaw,
                                 'collision': collision, 'distance_m': distance})
            if collision:
                r['status'] = 'collision'
            elif distance <= r['arrival_radius_m']:
                r['status'] = 'arrived'
            elif r['elapsed_s'] >= r['timeout_s']:
                r['status'] = 'timeout'
        return recent

    def recent(self, now=None):
        now = (self.history[-1][0] if self.history else time.monotonic()) if now is None else now
        history = [p for p in self.history if now-p[0] <= 8]
        if len(history) < 2:
            return {'window_s': 0, 'moved_m': 0, 'turned_deg': 0, 'target_closer_m': 0, 'pattern': 'starting'}
        moved = sum(math.dist(a[1], b[1]) for a,b in zip(history, history[1:]))
        turned = sum(abs((b[2]-a[2]+180)%360-180) for a,b in zip(history, history[1:]))
        progress = history[0][3]-history[-1][3]
        duration = history[-1][0]-history[0][0]
        pattern = ('starting' if duration < 3 else 'stuck' if moved < 0.1 and turned < 15
                   else 'turning_on_the_spot' if moved < 0.1 else 'advancing' if progress > 0.1
                   else 'moving_without_getting_closer')
        return dict(window_s=round(duration,2), moved_m=round(moved,3), turned_deg=round(turned,1),
                    target_closer_m=round(progress,3), pattern=pattern)

    def ray(self, xy, angle, config, mode, limit=5):
        last = 0.0
        for distance in np.arange(0.2, limit+0.01, 0.1):
            x, y = xy[0]+math.cos(angle)*distance, xy[1]+math.sin(angle)*distance
            if mode == 'observed':
                state = self.cells.get((math.floor(x/self.resolution), math.floor(y/self.resolution)), 'unknown')
            elif config.get('map_path'):
                state = 'unknown'  # Imported volumes require a separate body-band adapter.
            else:
                state = 'clear'
                ground = config['camera'].get('ground_z', 0)
                band = config['robot'].get('obstacle', {})
                for obj in config.get('objects', []):
                    cx,cy,cz = obj['center']; sx,sy,sz = obj['size']
                    if (abs(x-cx)<=sx/2 and abs(y-cy)<=sy/2 and
                        cz+sz/2>=max(ground+0.05, config['camera'].get('mount_height',0.45)+band.get('robot_z_bottom',-0.4)) and
                        cz-sz/2<=config['camera'].get('mount_height',0.45)+band.get('robot_z_top',0.4)):
                        state = 'blocked'; break
            if state != 'clear':
                return {'known_clear_distance_m': round(last,2), 'ends_in': state}
            last = float(distance)
        return {'known_clear_distance_m': round(last,2), 'ends_in': 'range_limit'}

    def world_state(self, xy, yaw, config, mode='observed'):
        target = config['target']; dx,dy = target[0]-xy[0], target[1]-xy[1]
        distance = math.hypot(dx,dy)
        bearing = (math.degrees(math.atan2(dy,dx))-yaw+180)%360-180
        directions = {name:self.ray(xy, math.radians(yaw+angle), config, mode)
                      for name,angle in DIRECTIONS}
        direct = self.ray(xy, math.atan2(dy,dx), config, mode, min(5,max(0.2,distance)))
        return {'schema_version': 1, 'source': mode, 'frame': 'robot_relative',
                'goal': {'bearing': bearing_name(bearing), 'bearing_deg': round(bearing,1),
                         'distance_m': round(distance,2), 'distance': 'near' if distance<2 else 'far',
                         'position_source': 'user_supplied_goal'},
                'robot': {'recent': self.recent()}, 'directions': directions,
                'way_to_target': direct,
                'notes': ['Distances are sampled center rays, not footprint-safe paths.',
                          'Unknown is not free. No semantic object labels are inferred.',
                          'Observed cells retain prior depth observations until reset.',
                          'Full-scene mode uses synthetic boxes only; imported maps are unknown.']}

    def metrics(self):
        if not self.run:
            return {'status': 'idle'}
        return {k:v for k,v in self.run.items() if k!='config'}

    def export(self):
        return {'schema_version':1, 'metrics': self.metrics(),
                'config': copy.deepcopy(self.run['config']) if self.run else None,
                'samples': copy.deepcopy(self.samples),
                'definitions': {'arrival': 'XY distance <= 0.35 m; no semantic/visibility check',
                                'collision': 'existing simulator geometric collision latch',
                                'stuck': '8 s history; >=3 s, <0.1 m movement and <15 deg turning',
                                'time': 'monotonic wall time; not deterministic simulation time'}}

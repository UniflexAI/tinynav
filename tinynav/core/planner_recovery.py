"""Observed planner recovery with bounded actions and map-aware route ranking."""
import math
import heapq
from collections import deque
import copy
import hashlib
import json
import time
import uuid
import numpy as np
from dataclasses import asdict
from types import SimpleNamespace


def footprint_cells(xy, robot, resolution):
    radius = math.hypot(robot.get('length',.4)/2+abs(robot.get('control_x',0)),
                        robot.get('width',.3)/2+abs(robot.get('control_y',0)))
    if robot.get('shape')=='circle':radius=max(radius,robot['radius'])
    radius += robot.get('recovery_margin',0)
    n = math.ceil(radius/resolution)
    return {(math.floor((xy[0]+i*resolution)/resolution),math.floor((xy[1]+j*resolution)/resolution))
            for i in range(-n,n+1) for j in range(-n,n+1) if math.hypot(i*resolution,j*resolution)<=radius+resolution/2}


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
        self.cell_observed_at = {}
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
        points = np.column_stack(((u-camera.get('cx',(w-1)/2))/camera['fx']*z,
                                  (v-camera.get('cy',(h-1)/2))/camera['fy']*z, z))
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
        observed_now = time.monotonic()
        for cell in free:
            if self.cells.get(cell) != 'blocked':
                self.cells[cell] = 'clear'
                self.cell_observed_at[cell] = observed_now
        for cell in occupied:
            self.cells[cell] = 'blocked'
            self.cell_observed_at[cell] = observed_now
        if len(self.cells) > 50000:
            center = origin[:2]/self.resolution
            self.cells = {k: val for k, val in self.cells.items() if math.dist(k, center) < 120}
            self.cell_observed_at = {k:v for k,v in self.cell_observed_at.items() if k in self.cells}

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


def advance(xy, yaw, v, w, dt):
    angle = math.radians(yaw)
    return [xy[0]+math.cos(angle)*v*dt, xy[1]+math.sin(angle)*v*dt], (yaw+math.degrees(w*dt)+180)%360-180


def poses(xy, yaw, stages):
    yield list(xy), yaw
    for stage in stages:
        remaining = stage['duration_s']
        while remaining > 1e-8:
            dt = min(.05, remaining)
            xy, yaw = advance(xy, yaw, stage['linear_mps'], stage['yaw_radps'], dt)
            yield xy, yaw
            remaining -= dt


def blocked_sweep(xy, yaw, stages, collision):
    return any(collision(p, a) for p, a in poses(xy, yaw, stages))


def proposals(xy, yaw, target, robot, cells, resolution, memory):
    v = min(.3, robot['max_linear_vel']); w = min(.6, robot['max_angular_vel'])
    if v <= 0 or w <= 0:return []
    radius = math.hypot(robot.get('length', .4)/2, robot.get('width', .3)/2)
    if robot.get('shape') == 'circle':radius = robot['radius']
    offsets = [(i*resolution, j*resolution) for i in range(-math.ceil(radius/resolution), math.ceil(radius/resolution)+1)
               for j in range(-math.ceil(radius/resolution), math.ceil(radius/resolution)+1) if math.hypot(i*resolution,j*resolution) <= radius+resolution/2]
    templates = []
    for side, sign in [('left',1),('right',-1)]:
        templates.append(('pivot_'+side, 0, 60, sign))
        for retreat, angle in [(.6,90),(1.2,90),(1.8,135)]:
            templates.append(('retreat_'+str(int(retreat*100))+'_'+side, retreat, angle, sign))
    if robot.get('recovery_detour',False):
        templates.extend([('retreat_300_left',3.0,90,1),('retreat_300_right',3.0,90,-1)])
    result = []
    for ident, retreat, angle, sign in templates:
        repeated = any(m['strategy_id']==ident and math.dist(xy,m['start_xy'])<.6 and abs((yaw-m['start_yaw_deg']+180)%360-180)<35 for m in memory)
        if repeated:continue
        stages = []
        if retreat:stages.append(dict(name='retreat',linear_mps=-v,yaw_radps=0,duration_s=retreat/v))
        stages.extend([dict(name='turn',linear_mps=0,yaw_radps=sign*w,duration_s=math.radians(angle)/w),
                       dict(name='probe',linear_mps=v,yaw_radps=0,duration_s=(1.8 if retreat==3.0 else .6)/v)])
        sampled = list(poses(xy,yaw,stages)); seen = set()
        for p, _ in sampled:
            seen.update((math.floor((p[0]+dx)/resolution),math.floor((p[1]+dy)/resolution)) for dx,dy in offsets)
        blocked = sum(cells.get(c)=='blocked' for c in seen)
        if blocked:continue
        unknown = sum(c not in cells for c in seen)
        end_xy, end_yaw = sampled[-1]
        result.append({'id':ident,'stages':stages,'duration_s':round(sum(s['duration_s'] for s in stages),2),
                       'predicted_goal_progress_m':round(math.dist(xy,target[:2])-math.dist(end_xy,target[:2]),3),
                       'displacement_m':round(math.dist(xy,end_xy),3),'final_yaw_deg':round(end_yaw,1),
                       'unknown_fraction':round(unknown/max(1,len(seen)),3),'observed_blocked_cells':blocked,
                       'observation_note':'Measured occupancy only. Unknown footprint cells require execution checks; no safety guarantee.'})
    return result


class RecoveryExecutor:
    def __init__(self):
        self.active = None
        self.memory = []
        self.events = []
        self.settle_until = 0

    def start(self, plan, xy, yaw, target, now):
        if self.active:raise ValueError('Recovery is already active')
        self.active = {'plan':copy.deepcopy(plan),'stage':0,'remaining':plan['stages'][0]['duration_s'],
                       'started':now,'start_xy':list(xy),'start_yaw_deg':yaw,'start_goal_m':math.dist(xy,target[:2]),'checked':False}
        self.events.append({'event':'strategy_started','strategy_id':plan['id']})

    def finish(self, reason, xy, target, now):
        if not self.active:return
        a = self.active
        outcome = {'strategy_id':a['plan']['id'],'start_xy':a['start_xy'],'start_yaw_deg':a['start_yaw_deg'],
                   'outcome':reason,'duration_s':round(now-a['started'],3),'end_xy':list(xy),
                   'goal_progress_m':round(a['start_goal_m']-math.dist(xy,target[:2]),3),
                   'retreat_m':sum(max(0,-s['linear_mps'])*s['duration_s'] for s in a['plan']['stages'])}
        self.memory.append(outcome);self.memory = self.memory[-24:]
        self.events.append({'event':'strategy_finished',**outcome});self.active = None;self.settle_until = now+5

    def step(self, xy, yaw, target, dt, now, report_age, collision):
        if not self.active:return None
        a = self.active
        if now-a['started'] > a['plan']['duration_s']+5:
            self.finish('execution_timeout',xy,target,now);return (0,0,dt)
        if report_age > .5:
            self.finish('stale_plan',xy,target,now);return (0,0,dt)
        stage = a['plan']['stages'][a['stage']]
        if not a['checked']:
            if blocked_sweep(xy,yaw,[{**stage,'duration_s':a['remaining']}],collision):
                self.finish('stage_collision_rejected',xy,target,now);return (0,0,dt)
            a['checked'] = True
            self.events.append({'event':'stage_started','strategy_id':a['plan']['id'],'stage':stage['name']})
        used = min(dt,a['remaining'])
        if blocked_sweep(xy,yaw,[{**stage,'duration_s':used}],collision):
            self.finish('collision_risk',xy,target,now);return (0,0,dt)
        a['remaining'] -= used
        if a['remaining'] <= 1e-8:
            a['stage'] += 1;a['checked'] = False
            if a['stage']==len(a['plan']['stages']):
                end_xy,_ = advance(xy,yaw,stage['linear_mps'],stage['yaw_radps'],used)
                self.finish('completed',end_xy,target,now)
            else:a['remaining'] = a['plan']['stages'][a['stage']]['duration_s']
        return stage['linear_mps'],stage['yaw_radps'],used


def shortlist(plans, xy, yaw, robot, cells, resolution):
    eligible = []
    excluded = {}
    for plan in plans:
        if not plan['id'].startswith('retreat_'):
            excluded[plan['id']] = 'not_a_retreat'
            continue
        swept = set()
        for point, _ in poses(xy,yaw,plan['stages']):
            swept.update(footprint_cells(point,robot,resolution))
        if any(cells.get(c) != 'clear' for c in swept):
            excluded[plan['id']] = 'unknown_or_blocked_footprint'
            continue
        retreat = sum(max(0,-s['linear_mps'])*s['duration_s'] for s in plan['stages'])
        if retreat <= 0:
            excluded[plan['id']] = 'no_reverse_displacement'
            continue
        eligible.append((retreat,plan))
    return {'plans':[p for _,p in eligible],'excluded':excluded,'rule':'observed_route_cost',
            'current_safety_unproven':True,'model_preference':None}


def route_cost(plan, previous_attempts=()):
    stages = plan['stages']
    retreat = sum(max(0,-s['linear_mps'])*s['duration_s'] for s in stages)
    length = sum(abs(s['linear_mps'])*s['duration_s'] for s in stages)
    duration = sum(s['duration_s'] for s in stages)
    progress = plan.get('predicted_goal_progress_m',0)
    failed = [a for a in previous_attempts if a.get('outcome')=='reblocked']
    minimum = max(1.2 if any(a['strategy_id'].startswith('pivot_') for a in failed) else 0,
                  2*max((a.get('retreat_m',0) for a in failed),default=0))
    escalation = 20.0 if minimum and retreat+1e-6 < minimum else 0.0
    repeats = sum(a.get('strategy_id')==plan['id'] for a in previous_attempts)
    remaining = plan.get('handoff_cost_m',-1.5*progress)
    return round(length+.05*duration+remaining+5*repeats+escalation,6)


def choose(screening, previous_attempts=()):
    plans = screening['plans']
    if not plans:return None
    return min(plans,key=lambda p:(route_cost(p,previous_attempts),p['id']))



def estimate_handoff(plans, xy, yaw, target, robot, cells, resolution):
    # Unknown cells may inform ranking, but never pass the execution coverage check.
    endpoints = {p['id']:list(poses(xy,yaw,p['stages']))[-1][0] for p in plans}
    if not endpoints:return
    def cell(point):return tuple(math.floor(v/resolution) for v in point[:2])
    goal = cell(target);points = [cell(xy),goal]+[cell(p) for p in endpoints.values()]
    pad = math.ceil(3/resolution)
    lo = [min(p[i] for p in points)-pad for i in (0,1)]
    hi = [max(p[i] for p in points)+pad for i in (0,1)]
    if (hi[0]-lo[0])*(hi[1]-lo[1])>40000:return
    offsets = footprint_cells([resolution*.5,resolution*.5],robot,resolution)
    blocked = {(x-dx,y-dy) for (x,y),value in cells.items() if value=='blocked' for dx,dy in offsets
               if lo[0]<=x-dx<=hi[0] and lo[1]<=y-dy<=hi[1]}
    if goal in blocked:return
    distances = {goal:0.0};queue = [(0.0,goal)];following = {}
    wanted = {cell(p) for p in endpoints.values()}
    neighbors = ((1,0),(-1,0),(0,1),(0,-1))
    while queue and wanted:
        cost,point = heapq.heappop(queue)
        if cost!=distances[point]:continue
        wanted.discard(point)
        for dx,dy in neighbors:
            other = (point[0]+dx,point[1]+dy)
            if other in blocked or not (lo[0]<=other[0]<=hi[0] and lo[1]<=other[1]<=hi[1]):continue
            step = resolution*(1 if cells.get(point)=='clear' else 1.5)
            proposed = cost+step
            if proposed < distances.get(other,float('inf')):
                distances[other]=proposed;following[other]=point
                heapq.heappush(queue,(proposed,other))
    for plan in plans:
        end = endpoints[plan['id']];point=cell(end)
        remaining = distances.get(point)
        if remaining is None:
            plan['handoff_cost_m']=100.0
            continue
        next_point = following.get(point,point)
        towards = sum((next_point[i]-point[i])*(target[i]-end[i]) for i in (0,1))
        # A greedy planner is likely to re-enter the trap if the required route starts backwards.
        plan['handoff_cost_m']=round(remaining+(6.0 if towards < -1e-6 else 0.0),3)


def recovery_candidates(offered,xy,yaw,robot,cells,resolution):
    coverage = {}
    pivots = []
    for plan in offered:
        swept = set()
        for point,_ in poses(xy,yaw,plan['stages']):
            swept.update(footprint_cells(point,robot,resolution))
        coverage[plan['id']] = {'footprint_cells':len(swept),
            'unknown_cells':sum(c not in cells for c in swept),
            'blocked_cells':sum(cells.get(c)=='blocked' for c in swept),
            'predicted_goal_progress_m':plan.get('predicted_goal_progress_m')}
        row = coverage[plan['id']]
        if plan['id'].startswith('pivot_') and plan.get('predicted_goal_progress_m',0)>.1 and row['unknown_cells']==row['blocked_cells']==0:
            pivots.append(plan)
    screened = shortlist(offered,xy,yaw,robot,cells,resolution)
    screened['plans'].extend(pivots)
    for plan in pivots:screened['excluded'].pop(plan['id'],None)
    for identity in screened['excluded']:
        if coverage[identity]['unknown_cells'] or coverage[identity]['blocked_cells']:
            screened['excluded'][identity] = 'unknown_or_blocked_footprint'
        elif identity.startswith('pivot_') and (coverage[identity]['predicted_goal_progress_m'] or 0)<=.1:
            screened['excluded'][identity] = 'insufficient_predicted_goal_progress'
    screened['coverage'] = coverage
    return screened


class RuleRecovery:
    def __init__(self):
        self.phase = 'watching'
        self.scan = None
        self.last_selection = None
        self.attempts = 0
        self.scans = 0
        self.next_scan = 0

    def status(self):
        return {'phase':self.phase,'scan':self.scan,'selection':self.last_selection,
                'attempts':self.attempts,'scan_attempts':self.scans,'decision_source':'rule','model_used':False}

    def step(self,node,now):
        def event(name,**values):
            node.experiment_events.append({'t':node.lab.metrics()['elapsed_s'],'event':name,'source':'rule',**values})
        def release(phase):
            node.override = None
            node.override_until = 0
            self.scan = None
            self.phase = phase
            self.next_scan = now+8
        running = node.running and node.lab.run and node.lab.run['status']=='running'
        if not running:
            if self.scan:
                release('stopped')
            self.phase = node.lab.run['status'] if node.lab.run else 'idle'
            return
        if node.world_mode != 'observed' or node.config.get('map_path'):
            if self.scan:
                release('unsupported_scene')
            self.phase = 'unsupported_scene'
            return
        age = now-node.plan_received_at if node.plan_received_at is not None else float('inf')
        if self.scan:
            scan = self.scan
            if node.collision or math.dist(scan['anchor_xy'],node.control_xy)>.02 or age>.5 or now-scan['started']>25:
                event('scan_aborted',reason='collision_drift_stale_report_or_timeout')
                release('scan_aborted')
                return
            scan['rotation_deg'] += abs((node.yaw_deg-scan['last_yaw']+180)%360-180)
            scan['last_yaw'] = node.yaw_deg
            if scan['rotation_deg'] < 359.5:
                node.override['yaw_radps'] = min(.6,node.config['robot']['max_angular_vel'],math.radians(360-scan['rotation_deg'])/.25)
                return
            node.override['yaw_radps'] = 0
            if self.phase != 'fresh_report':
                self.phase = 'fresh_report'
                scan['finished'] = now
                event('scan_completed',rotation_deg=scan['rotation_deg'])
                return
            if node.plan_received_at <= scan['finished']:
                return
            offered = proposals(node.control_xy,node.yaw_deg,node.config['target'],node.config['robot'],node.lab.cells,node.lab.resolution,node.recovery.memory)
            screened = recovery_candidates(offered,node.control_xy,node.yaw_deg,node.config['robot'],node.lab.cells,node.lab.resolution)
            memory = [a for a in node.recovery.memory if min(math.dist(node.control_xy,a['start_xy']),
                math.dist(node.control_xy,a.get('end_xy',a['start_xy'])))<1.0]
            estimate_handoff(screened['plans'],node.control_xy,node.yaw_deg,node.config['target'],
                node.config['robot'],node.lab.cells,node.lab.resolution)
            plan = choose(screened,memory)
            self.last_selection = {'eligible_ids':[p['id'] for p in screened['plans']],
                                   'excluded':screened['excluded'],'coverage':screened['coverage'],'rule':screened['rule'],
                                   'selected_id':plan['id'] if plan else None,'model_preference':None,
                                   'costs':{p['id']:route_cost(p,memory) for p in screened['plans']},
                                   'handoff_costs':{p['id']:p.get('handoff_cost_m') for p in screened['plans']}}
            event('candidates_screened',**self.last_selection)
            release('executing' if plan else 'no_observed_candidate')
            if plan:
                node.recovery.start(plan,node.control_xy,node.yaw_deg,node.config['target'],now)
                self.attempts += 1
            return
        if node.recovery.active:
            self.phase = 'executing'
            return
        if now < node.recovery.settle_until or now < self.next_scan:
            self.phase = 'planner_resumed'
            return
        self.phase = 'watching'
        recent = node.lab.recent()
        if self.scans >= 3:
            self.phase = 'attempt_limit'
            return
        if age>.5 or node.override or recent['window_s']<6 or recent['moved_m']>=.1:
            return
        if node.recovery.memory:
            last = node.recovery.memory[-1]
            if last.get('outcome')=='completed' and min(math.dist(node.control_xy,last.get('end_xy',node.control_xy)),
                    math.dist(node.control_xy,last['start_xy']))<.8:
                last['outcome']='reblocked'
                event('recovery_reblocked',strategy_id=last['strategy_id'],retreat_m=last.get('retreat_m',0))
        self.scan = {'anchor_xy':list(node.control_xy),'last_yaw':node.yaw_deg,'rotation_deg':0.0,'started':now}
        self.scans += 1
        self.phase = 'scanning'
        node.override = {'linear_mps':0,'yaw_radps':min(.6,node.config['robot']['max_angular_vel'])}
        node.override_until = now+26
        event('scan_started',trigger=recent)


class RecoveryRuntime:
    def __init__(self, robot):
        self.robot = asdict(robot)
        self.robot['max_linear_vel'] = min(.2,self.robot['max_linear_vel'])
        self.robot['max_angular_vel'] = min(.4,self.robot['max_angular_vel'])
        self.robot['recovery_margin'] = .05
        self.robot['recovery_detour'] = True
        self.lab = NavigationLab()
        self.policy = RuleRecovery()
        self.executor = RecoveryExecutor()
        self.target = None
        self.depth_at = None
        self.last_tick = None
        self.phase = 'idle'
        self.events = []
        self.started = None
        self.expected_pose = None
        self.goal_reached = False;self.override = None;self.last_xy = None;self.approach_active=False;self.approach_attempted=False;self.approach_started=None

    def reset(self, target=None):
        self.lab.reset();self.policy = RuleRecovery();self.executor = RecoveryExecutor()
        self.target = list(target) if target is not None else None
        self.last_tick = None;self.started = None;self.expected_pose = None
        self.goal_reached = False;self.override = None;self.last_xy = None;self.approach_active=False;self.approach_attempted=False;self.approach_started=None;self.phase = 'idle';self.events = [];self.depth_at = None

    def observe(self, depth, transform, camera, now):
        self.lab.observe_depth(depth,transform,camera,self.robot)
        self.depth_at = now
        # Clear evidence expires; unknown never becomes free from pose history.
        self.lab.cells = {c:v for c,v in self.lab.cells.items()
                          if now-self.lab.cell_observed_at[c]<=30 or v=='blocked'}
        self.lab.cell_observed_at = {c:t for c,t in self.lab.cell_observed_at.items() if c in self.lab.cells}

    def collision(self, xy, yaw):
        return any(self.lab.cells.get(c)!='clear' for c in footprint_cells(xy,self.robot,self.lab.resolution))

    def stop(self, reason):
        owned = self.policy.scan is not None or self.executor.active is not None or self.approach_active
        if self.executor.active and self.target is not None:
            self.executor.finish(reason,self.last_xy or self.executor.active['start_xy'],self.target,self.last_tick or 0)
        if owned:self.events.append({'event':'recovery_stopped','reason':reason,'source':'rule'})
        self.policy.scan = None;self.executor.active = None;self.expected_pose = None;self.override = None
        self.lab.history.clear();self.policy.next_scan = (self.last_tick or 0)+8
        self.phase = reason;self.approach_active=False
        return (0,0) if owned else None

    def tick(self, xy, yaw, now, active, paused, pose_age):
        if pose_age<=.5:self.last_xy=list(xy)
        dt = now-self.last_tick if self.last_tick is not None else 0
        self.last_tick = now
        if not active or paused or self.target is None:
            return self.stop('paused' if paused else 'inactive')
        if self.goal_reached:
            self.stop('arrived');return (0,0)
        if pose_age>.5 or self.depth_at is None or now-self.depth_at>.5:
            return self.stop('stale_sensor')
        if math.dist(xy,self.target[:2])<=.35:
            self.goal_reached = True;self.stop('arrived');return (0,0)
        if dt>.25:
            return self.stop('timer_gap')
        if self.expected_pose is not None:
            p,a=self.expected_pose
            if math.dist(xy,p)>.25 or abs((yaw-a+180)%360-180)>20:
                return self.stop('execution_pose_deviation')
        self.lab.history.append((now,list(xy),yaw,math.dist(xy,self.target[:2])))
        if self.started is None:self.started=now
        recent=self.lab.recent(now)
        if self.approach_active or (not self.approach_attempted and not self.executor.active and
                math.dist(xy,self.target[:2])<=.65 and recent['window_s']>=6 and recent['moved_m']<.1):
            command=self.goal_approach(xy,yaw,now)
            if command is not None:return command
        # RuleRecovery accepts a small adapter; it never publishes velocity itself.
        self.lab.run = {'status':'running'}
        self.lab.metrics = lambda:{'elapsed_s':round(now-self.started,3)}
        node = SimpleNamespace(running=True,world_mode='observed',config={'target':self.target,'robot':self.robot},
            lab=self.lab,plan_received_at=self.depth_at,collision=False,control_xy=xy,yaw_deg=yaw,
            override=self.override,override_until=0,recovery=self.executor,experiment_events=self.events)
        owned_before = self.policy.scan is not None or self.executor.active is not None
        self.policy.step(node,now)
        self.override = node.override
        self.phase = self.policy.phase
        if self.policy.scan:
            if self.collision(xy,yaw):
                return self.stop('scan_footprint_unknown_or_blocked') or (0,0)
            self.expected_pose = None
            if not node.override:return (0,0)
            w=node.override['yaw_radps']
            if w:w=math.copysign(min(self.robot['max_angular_vel'],max(abs(w),self.robot['min_angular_vel'])),w)
            return node.override['linear_mps'],w
        if self.executor.active:
            stage = self.executor.active['plan']['stages'][self.executor.active['stage']]
            remaining = {**stage,'duration_s':self.executor.active['remaining']}
            # Recheck the entire remaining stage against the latest observed map.
            if any(self.collision(p,a) for p,a in poses(xy,yaw,[remaining])):
                self.executor.finish('updated_footprint_rejected',xy,self.target,now)
                self.expected_pose = None;return (0,0)
            command = self.executor.step(xy,yaw,self.target,max(0,dt),now,now-self.depth_at,self.collision)
            if command:
                v,w,_=command
                self.expected_pose = (list(xy),yaw) if self.expected_pose is None else self.expected_pose
                p,a=self.expected_pose
                self.expected_pose = (list(poses(p,a,[{'linear_mps':v,'yaw_radps':w,'duration_s':max(0,dt)}]))[-1])
                return v,w
        self.expected_pose = None
        return (0,0) if owned_before else None

    def goal_approach(self, xy, yaw, now):
        if self.approach_active and now-self.approach_started>15:
            return self.stop('goal_approach_timeout')
        if self.collision(xy,yaw):
            return (0,0) if self.approach_active else None
        bearing=math.atan2(self.target[1]-xy[1],self.target[0]-xy[0])
        error=(bearing-math.radians(yaw)+math.pi)%(2*math.pi)-math.pi
        v=0.0;w=0.0
        if abs(error)>.15:
            w=math.copysign(min(self.robot['max_angular_vel'],max(self.robot['min_angular_vel'],abs(error)*.8)),error)
        else:
            v=self.robot['min_linear_vel']
            if not 0<v<=self.robot['max_linear_vel']:return None
            distance=math.dist(xy,self.target[:2])-.35
            stage={'linear_mps':v,'yaw_radps':0,'duration_s':distance/v}
            if any(self.collision(p,a) for p,a in poses(xy,yaw,[stage])):
                return (0,0) if self.approach_active else None
        if not self.approach_active:
            self.policy.scan=None;self.override=None
            self.approach_active=True;self.approach_attempted=True;self.approach_started=now
            self.events.append({'event':'goal_approach_started','source':'rule'})
        self.phase='goal_approach'
        return v,w

    def status(self):
        return {'phase':self.phase,'enabled':True,'decision_source':'rule','model_used':False,
                'scan':self.policy.scan,'scan_attempts':self.policy.scans,'active_strategy':self.executor.active,'attempts':self.executor.memory,
                'selection':self.policy.last_selection,'events':(self.events+self.executor.events)[-40:]}

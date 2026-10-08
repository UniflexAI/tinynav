"""Rule-based planner recovery: bounded maneuvers ranked on a retained observed map.

The planner feeds depth frames via observe() and calls tick() at plan rate.
tick() returns (v, w) while recovery owns motion, else None.
Unknown space is never assumed free: only measured-clear footprints are used.
"""
import copy
import heapq
import math
from collections import deque
from dataclasses import asdict

import numpy as np


def footprint_cells(xy, robot, resolution):
    radius = math.hypot(robot.get('length',.4)/2+abs(robot.get('control_x',0)),
                        robot.get('width',.3)/2+abs(robot.get('control_y',0)))
    if robot.get('shape')=='circle':radius=max(radius,robot['radius'])
    radius += robot.get('recovery_margin',0)
    n = math.ceil(radius/resolution)
    return {(math.floor((xy[0]+i*resolution)/resolution),math.floor((xy[1]+j*resolution)/resolution))
            for i in range(-n,n+1) for j in range(-n,n+1) if math.hypot(i*resolution,j*resolution)<=radius+resolution/2}


class MotionTracker:
    """Shared motion history: anchor-stall (guide) and windowed path length (recovery)."""
    move_epsilon_m = .1
    window_s = 8.0

    def __init__(self, maxlen=600):
        self.history = deque(maxlen=maxlen)  # (now, [x, y])
        self.anchor = None
        self.anchor_since = None

    def reset(self):
        self.history.clear()
        self.anchor = None
        self.anchor_since = None

    def update(self, xy, now):
        xy = [float(xy[0]), float(xy[1])]
        if self.anchor is None or math.dist(xy, self.anchor) > self.move_epsilon_m:
            self.anchor, self.anchor_since = xy, now
        self.history.append((now, xy))

    def stalled(self, now, duration_s):
        """True when the robot stayed within move_epsilon_m for duration_s."""
        return self.anchor_since is not None and now - self.anchor_since >= duration_s

    def recent(self, now):
        """(window duration, path length) inside window_s, excluding stale gaps."""
        points = [p for p in self.history if now - p[0] <= self.window_s]
        if len(points) < 2:
            return 0.0, 0.0
        moved = sum(math.dist(a[1], b[1]) for a, b in zip(points, points[1:]))
        return points[-1][0] - points[0][0], moved


class ObservationMap:
    """Conservative local occupancy: measured clear/blocked cells at fixed resolution.

    Blocked evidence never expires (dead-end walls must stay); clear evidence
    expires so moved obstacles stop blocking after clear_ttl_s.
    """
    resolution = 0.1
    clear_ttl_s = 30.0
    max_cells = 50000

    def __init__(self):
        self.cells = {}
        self.observed_at = {}

    def reset(self):
        self.cells = {}
        self.observed_at = {}

    def observe(self, depth, transform, camera, robot, now):
        # Only measured rays carve free space. Zero/no-return depth stays unknown.
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
            ray = origin + np.arange(n)[:, None]/n*(endpoint-origin)
            ray = ray[(ray[:, 2] >= bottom) & (ray[:, 2] <= top)]
            free.update(map(tuple, np.floor(ray[:, :2]/self.resolution).astype(int)))
            if bottom <= endpoint[2] <= top:
                occupied.add(tuple(np.floor(endpoint[:2]/self.resolution).astype(int)))
        for cell in free:
            if self.cells.get(cell) != 'blocked':
                self.cells[cell] = 'clear'
                self.observed_at[cell] = now
        for cell in occupied:
            self.cells[cell] = 'blocked'
            self.observed_at[cell] = now
        # Clear evidence expires; blocked evidence is retained until reset.
        self.cells = {c: v for c, v in self.cells.items()
                      if v == 'blocked' or now - self.observed_at[c] <= self.clear_ttl_s}
        self.observed_at = {c: t for c, t in self.observed_at.items() if c in self.cells}
        if len(self.cells) > self.max_cells:
            center = origin[:2]/self.resolution
            self.cells = {k: v for k, v in self.cells.items() if math.dist(k, center) < 120}
            self.observed_at = {c: t for c, t in self.observed_at.items() if c in self.cells}

    def footprint_clear(self, xy, yaw, robot):
        """Unknown counts as blocked: only measured-clear footprints pass."""
        return all(self.cells.get(c) == 'clear' for c in footprint_cells(xy, robot, self.resolution))


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
                       'unknown_fraction':round(unknown/max(1,len(seen)),3),'observed_blocked_cells':blocked})
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
    return {'plans':[p for _,p in eligible],'excluded':excluded}


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


class RecoveryRuntime:
    """Scan-rank-execute recovery owned by the planner.

    Phases: watching -> scanning -> fresh_report -> executing -> planner_resumed,
    plus terminal guards (paused/inactive/arrived/stale_sensor/timer_gap/...).
    """
    goal_radius_m = .35
    approach_radius_m = .65
    approach_timeout_s = 15.0
    stale_s = .5
    timer_gap_s = .25
    drift_tolerance_m = .02
    pose_deviation_m = .25
    pose_deviation_deg = 20
    scan_rotation_deg = 359.5
    scan_timeout_s = 25.0
    scan_interval_s = 8.0
    max_scans = 3
    stall_window_s = 6.0
    stall_path_m = .1

    def __init__(self, robot, tracker=None):
        self.robot = asdict(robot)
        self.robot['max_linear_vel'] = min(.2, self.robot['max_linear_vel'])
        self.robot['max_angular_vel'] = min(.4, self.robot['max_angular_vel'])
        self.robot['recovery_margin'] = .05
        self.robot['recovery_detour'] = True
        self.map = ObservationMap()
        self.tracker = tracker if tracker is not None else MotionTracker()
        self.reset()

    def reset(self, target=None):
        self.map.reset()
        self.tracker.reset()
        self.executor = RecoveryExecutor()
        self.target = list(target) if target is not None else None
        self.phase = 'idle'
        self.scan = None
        self.last_selection = None
        self.attempts = 0
        self.scans = 0
        self.next_scan = 0.0
        self.depth_at = None
        self.last_tick = None
        self.last_xy = None
        self.expected_pose = None
        self.goal_reached = False
        self.approach_active = False
        self.approach_attempted = False
        self.approach_started = None
        self.events = []

    def owns_motion(self):
        return self.scan is not None or self.executor.active is not None or self.approach_active

    def observe(self, depth, transform, camera, now):
        self.map.observe(depth, transform, camera, self.robot, now)
        self.depth_at = now

    def stop(self, reason, now):
        owned = self.owns_motion()
        if self.executor.active and self.target is not None:
            self.executor.finish(reason, self.last_xy or self.executor.active['start_xy'], self.target, now)
        if owned:
            self.events.append({'event': 'recovery_stopped', 'reason': reason})
        self.scan = None
        self.executor.active = None
        self.expected_pose = None
        self.tracker.reset()
        self.next_scan = now + self.scan_interval_s
        self.phase = reason
        self.approach_active = False
        return (0, 0) if owned else None

    def tick(self, xy, yaw, now, active, paused, pose_age):
        if pose_age <= self.stale_s:
            self.last_xy = list(xy)
        dt = now - self.last_tick if self.last_tick is not None else 0
        self.last_tick = now
        if not active or paused or self.target is None:
            return self.stop('paused' if paused else 'inactive', now)
        if self.goal_reached:
            self.stop('arrived', now)
            return (0, 0)
        if pose_age > self.stale_s or self.depth_at is None or now - self.depth_at > self.stale_s:
            return self.stop('stale_sensor', now)
        if math.dist(xy, self.target[:2]) <= self.goal_radius_m:
            self.goal_reached = True
            self.stop('arrived', now)
            return (0, 0)
        if dt > self.timer_gap_s:
            return self.stop('timer_gap', now)
        if self.expected_pose is not None:
            p, a = self.expected_pose
            if math.dist(xy, p) > self.pose_deviation_m or abs((yaw-a+180)%360-180) > self.pose_deviation_deg:
                return self.stop('execution_pose_deviation', now)
        self.tracker.update(xy, now)
        window_s, moved_m = self.tracker.recent(now)
        if self.approach_active or (not self.approach_attempted and self.executor.active is None
                and math.dist(xy, self.target[:2]) <= self.approach_radius_m
                and window_s >= self.stall_window_s and moved_m < self.stall_path_m):
            command = self.goal_approach(xy, yaw, now)
            if command is not None:
                return command
        owned_before = self.owns_motion()
        command = self._recovery_step(xy, yaw, now, dt, window_s, moved_m)
        if command is not None:
            return command
        return (0, 0) if owned_before else None

    def goal_approach(self, xy, yaw, now):
        if self.approach_active and now - self.approach_started > self.approach_timeout_s:
            return self.stop('goal_approach_timeout', now)
        if not self.map.footprint_clear(xy, yaw, self.robot):
            return (0, 0) if self.approach_active else None
        bearing = math.atan2(self.target[1]-xy[1], self.target[0]-xy[0])
        error = (bearing-math.radians(yaw)+math.pi)%(2*math.pi)-math.pi
        v = w = 0.0
        if abs(error) > .15:
            w = math.copysign(min(self.robot['max_angular_vel'], max(self.robot['min_angular_vel'], abs(error)*.8)), error)
        else:
            v = self.robot['min_linear_vel']
            if not 0 < v <= self.robot['max_linear_vel']:
                return None
            distance = math.dist(xy, self.target[:2]) - self.goal_radius_m
            stage = {'linear_mps': v, 'yaw_radps': 0, 'duration_s': distance/v}
            if any(not self.map.footprint_clear(p, a, self.robot) for p, a in poses(xy, yaw, [stage])):
                return (0, 0) if self.approach_active else None
        if not self.approach_active:
            self.scan = None
            self.approach_active = True
            self.approach_attempted = True
            self.approach_started = now
            self.events.append({'event': 'goal_approach_started'})
        self.phase = 'goal_approach'
        return v, w

    def _release(self, phase, now):
        self.scan = None
        self.phase = phase
        self.next_scan = now + self.scan_interval_s

    def _recovery_step(self, xy, yaw, now, dt, window_s, moved_m):
        depth_age = now - self.depth_at
        # --- decision: scan state machine ---
        if self.scan is not None:
            self.expected_pose = None
            scan = self.scan
            if not self.map.footprint_clear(xy, yaw, self.robot):
                return self.stop('scan_footprint_unknown_or_blocked', now) or (0, 0)
            if (math.dist(scan['anchor_xy'], xy) > self.drift_tolerance_m
                    or depth_age > self.stale_s or now - scan['started'] > self.scan_timeout_s):
                self._release('scan_aborted', now)
            else:
                scan['rotation_deg'] += abs((yaw-scan['last_yaw']+180)%360-180)
                scan['last_yaw'] = yaw
                if scan['rotation_deg'] < self.scan_rotation_deg:
                    w = min(.6, self.robot['max_angular_vel'], math.radians(360-scan['rotation_deg'])/.25)
                    if w:
                        w = math.copysign(min(self.robot['max_angular_vel'], max(abs(w), self.robot['min_angular_vel'])), w)
                    return 0.0, w
                if self.phase != 'fresh_report':
                    self.phase = 'fresh_report'
                    scan['finished'] = now
                    self.events.append({'event': 'scan_completed', 'rotation_deg': round(scan['rotation_deg'], 1)})
                elif self.depth_at > scan['finished']:
                    self._select_and_start(xy, yaw, now)
        elif self.executor.active is not None:
            self.phase = 'executing'
        elif now < self.executor.settle_until or now < self.next_scan:
            self.phase = 'planner_resumed'
        else:
            self.phase = 'watching'
            if self.scans >= self.max_scans:
                self.phase = 'attempt_limit'
            elif window_s >= self.stall_window_s and moved_m < self.stall_path_m:
                if self.executor.memory:
                    last = self.executor.memory[-1]
                    if last.get('outcome') == 'completed' and min(math.dist(xy, last.get('end_xy', xy)),
                            math.dist(xy, last['start_xy'])) < .8:
                        last['outcome'] = 'reblocked'
                        self.events.append({'event': 'recovery_reblocked', 'strategy_id': last['strategy_id'],
                                            'retreat_m': last.get('retreat_m', 0)})
                self.scan = {'anchor_xy': list(xy), 'last_yaw': yaw, 'rotation_deg': 0.0, 'started': now}
                self.scans += 1
                self.phase = 'scanning'
                self.events.append({'event': 'scan_started'})
                if not self.map.footprint_clear(xy, yaw, self.robot):
                    return self.stop('scan_footprint_unknown_or_blocked', now) or (0, 0)
                w = min(.6, self.robot['max_angular_vel'])
                w = math.copysign(min(self.robot['max_angular_vel'], max(abs(w), self.robot['min_angular_vel'])), w)
                return 0.0, w
        # --- execution: fresh_report wait or active maneuver ---
        if self.scan is not None:
            return 0.0, 0.0
        if self.executor.active is not None:
            stage = self.executor.active['plan']['stages'][self.executor.active['stage']]
            remaining = {**stage, 'duration_s': self.executor.active['remaining']}
            # Recheck the entire remaining stage against the latest observed map.
            if any(not self.map.footprint_clear(p, a, self.robot) for p, a in poses(xy, yaw, [remaining])):
                self.executor.finish('updated_footprint_rejected', xy, self.target, now)
                self.expected_pose = None
                return (0.0, 0.0)
            collision = lambda p, a: not self.map.footprint_clear(p, a, self.robot)
            command = self.executor.step(xy, yaw, self.target, max(0, dt), now, depth_age, collision)
            if command:
                v, w, _ = command
                self.expected_pose = (list(xy), yaw) if self.expected_pose is None else self.expected_pose
                p, a = self.expected_pose
                self.expected_pose = list(poses(p, a, [{'linear_mps': v, 'yaw_radps': w, 'duration_s': max(0, dt)}]))[-1]
                return v, w
        self.expected_pose = None
        return None

    def _select_and_start(self, xy, yaw, now):
        cells, resolution = self.map.cells, self.map.resolution
        offered = proposals(xy, yaw, self.target, self.robot, cells, resolution, self.executor.memory)
        screened = recovery_candidates(offered, xy, yaw, self.robot, cells, resolution)
        memory = [a for a in self.executor.memory if min(math.dist(xy, a['start_xy']),
                math.dist(xy, a.get('end_xy', a['start_xy']))) < 1.0]
        estimate_handoff(screened['plans'], xy, yaw, self.target, self.robot, cells, resolution)
        plan = choose(screened, memory)
        self.last_selection = {'eligible_ids': [p['id'] for p in screened['plans']],
                               'excluded': screened['excluded'], 'coverage': screened['coverage'],
                               'selected_id': plan['id'] if plan else None,
                               'costs': {p['id']: route_cost(p, memory) for p in screened['plans']},
                               'handoff_costs': {p['id']: p.get('handoff_cost_m') for p in screened['plans']}}
        self._release('executing' if plan else 'no_observed_candidate', now)
        if plan:
            self.executor.start(plan, xy, yaw, self.target, now)
            self.attempts += 1

    def status(self):
        return {'phase': self.phase, 'decision_source': 'rule',
                'scan': self.scan, 'scan_attempts': self.scans, 'attempts': self.attempts,
                'active_strategy': self.executor.active, 'memory': self.executor.memory,
                'selection': self.last_selection,
                'events': (self.events + self.executor.events)[-40:]}

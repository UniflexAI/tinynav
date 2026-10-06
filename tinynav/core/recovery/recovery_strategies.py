"""Bounded recovery proposals from measured occupancy, with staged execution."""
import copy
import math


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
                   'goal_progress_m':round(a['start_goal_m']-math.dist(xy,target[:2]),3)}
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

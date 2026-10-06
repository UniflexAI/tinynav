"""Explicit rule-assisted observed recovery, driven under the simulation lock."""
import math
from tinynav.core.recovery.recovery_strategies import proposals, poses
from tinynav.core.recovery.geometry import footprint_cells
from tinynav.core.recovery.observed_retreat_selection import shortlist, choose


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
    if len(pivots)==1:
        selected = pivots[0]
        screened = {'plans':[selected],'excluded':{p['id']:'unique_observed_progress_pivot_preferred' for p in offered if p!=selected},
                    'rule':'unique_observed_progress_pivot','model_preference':None,'current_safety_unproven':True}
    elif not screened['plans'] and pivots:
        selected = max(pivots,key=lambda p:p.get('predicted_goal_progress_m',0))
        screened = {'plans':[selected],'excluded':{p['id']:'observed_progress_pivot_when_no_retreat' for p in offered if p!=selected},
                    'rule':'observed_progress_pivot_when_no_retreat','model_preference':None,'current_safety_unproven':True}
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
            plan = choose(screened,node.recovery.memory)
            self.last_selection = {'eligible_ids':[p['id'] for p in screened['plans']],
                                   'excluded':screened['excluded'],'coverage':screened['coverage'],'rule':screened['rule'],
                                   'selected_id':plan['id'] if plan else None,'model_preference':None}
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
        self.scan = {'anchor_xy':list(node.control_xy),'last_yaw':node.yaw_deg,'rotation_deg':0.0,'started':now}
        self.scans += 1
        self.phase = 'scanning'
        node.override = {'linear_mps':0,'yaw_radps':min(.6,node.config['robot']['max_angular_vel'])}
        node.override_until = now+26
        event('scan_started',trigger=recent)

"""Observed recovery runtime; one caller owns all velocity arbitration."""
import math
from dataclasses import asdict
from types import SimpleNamespace
from tinynav.core.recovery.observations import NavigationLab
from tinynav.core.recovery.rule_recovery import RuleRecovery
from tinynav.core.recovery.recovery_strategies import RecoveryExecutor, poses
from tinynav.core.recovery.geometry import footprint_cells


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

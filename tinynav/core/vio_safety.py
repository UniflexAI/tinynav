"""VIO fault latch shared by ROS supervision and deterministic offline tests."""
import json
import math
import os
from pathlib import Path
import uuid


class VioSafety:
    pose_timeout = .5
    depth_timeout = .75
    status_timeout = 3.
    settle_s = 2.

    def __init__(self, state_path=None):
        self.path = Path(state_path) if state_path else None
        saved = json.loads(self.path.read_text()) if self.path and self.path.exists() else {}
        self.session = uuid.uuid4().hex
        self.epoch = int(saved.get('epoch', 0)) + 1
        self.fault_id = int(saved.get('fault_id', 0))
        self.locked = bool(saved.get('locked', False))
        self.reason = saved.get('reason', '') if self.locked else ''
        if saved and not self.locked:
            self.locked = True
            self.fault_id += 1
            self.reason = 'supervisor_restart'
        self.vio_required = bool(saved.get('vio_required', False))
        self.vio_status = None
        self.status_rx = None
        self.pose_rx = self.depth_rx = None
        self.pose_stamp = self.depth_stamp = None
        self.position = None
        self.accept_after = None
        self.quiet_until = -math.inf
        self.stale = set()
        self.had_input = False
        self.needs_settle = self.locked
        self._save()

    def _save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(epoch=self.epoch, fault_id=self.fault_id,
                locked=self.locked, reason=self.reason, vio_required=self.vio_required)))
            os.replace(temp, self.path)

    def fault(self, reason, now, hard=True, stamp=None):
        self.epoch += 1
        if hard:
            self.fault_id += 1
            self.locked = True
        self.reason = reason
        self.quiet_until = now + self.settle_s
        self.pose_rx = self.depth_rx = None
        self.accept_after = stamp if stamp is not None else self.pose_stamp
        self._save()

    def pose(self, stamp, position, now):
        if not math.isfinite(stamp) or not all(math.isfinite(x) for x in position):
            if 'nonfinite' not in self.stale:
                self.fault('nonfinite_pose', now)
                self.stale.add('nonfinite')
            return
        if self.needs_settle:
            self.quiet_until = now + self.settle_s
            self.needs_settle = False
        if self.pose_stamp is not None and stamp <= self.pose_stamp:
            if stamp < self.pose_stamp - 1.:
                self.fault('pose_clock_reset', now, stamp=stamp)
                self.pose_stamp = self.depth_stamp = None
            else:
                return
        if self.position is not None and self.pose_stamp is not None:
            step = math.dist(position, self.position)
            dt = stamp - self.pose_stamp
            if step >= 1.:
                self.fault('meter_jump', now, stamp=stamp)
            elif dt > 1. or (step > .1 and step > 3. * max(dt, .001)):
                self.fault('pose_jump', now, hard=False, stamp=stamp)
        self.had_input = True
        self.pose_stamp, self.position, self.pose_rx = stamp, tuple(position), now
        self.stale.discard('pose')
        self.stale.discard('nonfinite')

    def depth(self, stamp, now):
        if not math.isfinite(stamp) or (self.depth_stamp is not None and stamp <= self.depth_stamp):
            return
        if self.accept_after is not None and stamp < self.accept_after:
            return
        self.depth_stamp, self.depth_rx = stamp, now
        self.stale.discard('depth')

    def status(self, status, now):
        status = status.strip().upper()
        previous = self.vio_status
        first = not self.vio_required
        self.vio_required = True
        self.vio_status, self.status_rx = status, now
        if status not in ('TRACKING', 'TRACKING_STATIC'):
            if status != previous and (status == 'RESTARTING' or self.had_input):
                self.fault('vio_' + status.lower(), now)
        elif previous not in (None, 'TRACKING', 'TRACKING_STATIC'):
            self.quiet_until = now + self.settle_s
            self.pose_rx = self.depth_rx = None
        if first:
            self._save()
        self.stale.discard('status')

    def tick(self, now):
        if self.had_input:
            for name, rx, timeout in [('pose', self.pose_rx, self.pose_timeout),
                                       ('depth', self.depth_rx, self.depth_timeout),
                                       ('status', self.status_rx, self.status_timeout)]:
                if name == 'status' and not self.vio_required:
                    continue
                # A fault invalidates samples; wait for new samples rather than emitting another fault per tick.
                if rx is not None and now - rx > timeout and name not in self.stale:
                    self.stale.add(name)
                    self.fault(name + '_timeout', now)
                    break

    def snapshot(self, now):
        self.tick(now)
        fresh = self.pose_rx is not None and self.depth_rx is not None \
            and now - self.pose_rx <= self.pose_timeout and now - self.depth_rx <= self.depth_timeout
        vio_ok = not self.vio_required or (self.vio_status in ('TRACKING', 'TRACKING_STATIC')
            and self.status_rx is not None and now - self.status_rx <= self.status_timeout)
        healthy = bool(fresh and vio_ok and now >= self.quiet_until)
        return dict(session=self.session, epoch=self.epoch, fault_id=self.fault_id,
                    locked=self.locked, blocked=self.locked or not healthy,
                    ready=self.locked and healthy, reason=self.reason, accept_after=self.accept_after,
                    vio_status=self.vio_status, pose_age=None if self.pose_rx is None else now-self.pose_rx,
                    depth_age=None if self.depth_rx is None else now-self.depth_rx)

    def confirm(self, fault_id, now):
        state = self.snapshot(now)
        if not self.locked or fault_id != self.fault_id:
            return False, 'Fault changed; refresh the recovery status'
        if not state['ready']:
            return False, 'VIO and depth must be fresh and stable for 2 seconds'
        self.locked = False
        self.reason = ''
        self.epoch += 1
        self.accept_after = self.pose_stamp
        self._save()
        return True, 'VIO recovery confirmed'


def safety_blocked(state, received, now):
    return state is not None and (state.get('blocked', True) or now - received > 1.)


def accept_safety(state, current):
    if not current or 'epoch' not in current:
        return True
    return state['epoch'] > current['epoch'] or (state['epoch'] == current['epoch']
        and state['session'] == current['session'])

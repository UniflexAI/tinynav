"""One bounded, clearance-checked forward start for an unresponsive Go2."""
from time import monotonic

import numpy as np

from tinynav.rl.bev import GRID_RES
from tinynav.rl.rl_planner import footprint_clearance

START_SPEED = .25
STALL_SECONDS = .8
PULSE_SECONDS = .55
MAX_COMMAND_GAP = .25
MAX_POSE_AGE = .50


class SemanticStartAssist:
    def __init__(self, *, checked_start=False):
        self.checked_start = checked_start
        self.reset()

    def reset(self):
        self.command = None
        self.anchor = None
        self.attempted = False
        self.pulse_until = 0.
        self.sustaining = False

    def observe_command(self, v, w):
        now = monotonic()
        previous = self.command
        low_since = None
        if .08 <= v <= .20 and abs(w) <= .08:
            low_since = now
            if previous is not None and previous[3] is not None and now - previous[0] <= MAX_COMMAND_GAP:
                low_since = previous[3]
        self.command = (now, v, w, low_since)

    def forward(self, center, yaw, selected, guide_turn, cap, age, esdf, origin, edge, *, goal=None, velocity=None):
        now = monotonic()
        sample = self.command
        max_age = .75 if self.checked_start else MAX_POSE_AGE
        if sample is None or now - sample[0] > MAX_COMMAND_GAP or not 0 <= age <= max_age:
            self.anchor = None
            self.pulse_until = 0.
            self.sustaining = False
            return None
        if self.checked_start:
            delta = np.asarray(goal) - center
            heading = np.angle(np.exp(1j * (np.arctan2(delta[1], delta[0]) - yaw)))
            if np.linalg.norm(delta) < .75 or abs(heading) > np.deg2rad(20.):
                self.anchor = None
                self.pulse_until = 0.
                self.sustaining = False
                return None
        if self.anchor is not None:
            _, position, heading = self.anchor
            moved = (np.linalg.norm(center - position) >= .025
                     or abs(np.angle(np.exp(1j * (yaw-heading)))) >= .055)
            if moved:
                forward = np.array([np.cos(heading), np.sin(heading)])
                forward_progress = (center - position) @ forward >= .025
                if self.checked_start and self.attempted and forward_progress and abs(
                    np.angle(np.exp(1j * (yaw - heading)))
                ) < .055:
                    self.anchor = (now, center.copy(), yaw)
                    self.sustaining = True
                else:
                    self.anchor = None
                    self.attempted = False
                    self.pulse_until = 0.
                    self.sustaining = False
                    return None
        if not (.08 <= selected[0] <= .20 and abs(selected[1]) <= .10
                and abs(guide_turn) <= .25 and cap >= START_SPEED):
            self.anchor = None
            self.pulse_until = 0.
            self.sustaining = False
            return None

        low = .08 <= sample[1] <= .20 and abs(sample[2]) <= .08
        if self.sustaining:
            if self.anchor is None or now - self.anchor[0] > STALL_SECONDS:
                self.sustaining = False
                self.pulse_until = 0.
                return None
            if abs(sample[2]) > .08 or not .08 <= sample[1] <= START_SPEED + .01:
                self.sustaining = False
                self.pulse_until = 0.
                return None
        elif self.pulse_until <= now:
            self.pulse_until = 0.
            if not low:
                self.anchor = None
                return None
            if self.anchor is None:
                self.anchor = (now, center.copy(), yaw)
            if self.attempted or now - max(self.anchor[0], sample[3]) < STALL_SECONDS:
                return None
        elif abs(sample[2]) > .08 or not .08 <= sample[1] <= START_SPEED + .01:
            self.pulse_until = 0.
            return None

        if self.checked_start:
            from tinynav.rl.semantic_forward_guide import candidate_path, has_clearance

            # Check travel before the new command takes effect as well as the
            # forward pulse and slow braking after it.
            pose = np.r_[center, yaw]
            if not np.isfinite(velocity).all() or not all(
                has_clearance(candidate_path(pose, prior, 0., delay), esdf, origin, edge)
                for prior in (sample[1:3], velocity, (0., 0.))
                for delay in (.32, max(.50, age))
            ):
                self.pulse_until = 0.
                self.sustaining = False
                return None

        # Use instant full response, controller lease, and slow braking. Never
        # credit the dead zone when checking how far the chassis can travel.
        distance = START_SPEED * 1.50 + START_SPEED**2 / (2 * .30)
        x = np.linspace(0., distance, 1 + int(np.ceil(distance / .0125)))
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        sweep = np.column_stack((center + x[:, None] * forward, np.full(len(x), yaw)))
        room = footprint_clearance(sweep, esdf, origin, GRID_RES, edge=edge)
        if not np.isfinite(room).all() or room.min() < .10:
            self.pulse_until = 0.
            self.sustaining = False
            return None
        if not self.attempted:
            self.pulse_until = now + (1. if self.checked_start else PULSE_SECONDS)
            self.attempted = True
        return START_SPEED, 0.

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
    def __init__(self):
        self.reset()

    def reset(self):
        self.command = None
        self.anchor = None
        self.attempted = False
        self.pulse_until = 0.

    def observe_command(self, v, w):
        now = monotonic()
        previous = self.command
        low_since = None
        if .08 <= v <= .20 and abs(w) <= .08:
            low_since = now
            if previous is not None and previous[3] is not None and now - previous[0] <= MAX_COMMAND_GAP:
                low_since = previous[3]
        self.command = (now, v, w, low_since)

    def forward(self, center, yaw, selected, guide_turn, cap, age, esdf, origin, edge):
        now = monotonic()
        sample = self.command
        if sample is None or now - sample[0] > MAX_COMMAND_GAP or not 0 <= age <= MAX_POSE_AGE:
            self.anchor = None
            self.pulse_until = 0.
            return None
        if self.anchor is not None:
            _, position, heading = self.anchor
            moved = (np.linalg.norm(center - position) >= .025
                     or abs(np.angle(np.exp(1j * (yaw-heading)))) >= .055)
            if moved:
                self.anchor = None
                self.attempted = False
                self.pulse_until = 0.
                return None
        if not (.08 <= selected[0] <= .20 and abs(selected[1]) <= .10
                and abs(guide_turn) <= .25 and cap >= START_SPEED):
            self.anchor = None
            self.pulse_until = 0.
            return None

        low = .08 <= sample[1] <= .20 and abs(sample[2]) <= .08
        if self.pulse_until <= now:
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

        # Use instant full response, controller lease, and slow braking. Never
        # credit the dead zone when checking how far the chassis can travel.
        distance = START_SPEED * 1.50 + START_SPEED**2 / (2 * .30)
        x = np.linspace(0., distance, 1 + int(np.ceil(distance / .0125)))
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        sweep = np.column_stack((center + x[:, None] * forward, np.full(len(x), yaw)))
        room = footprint_clearance(sweep, esdf, origin, GRID_RES, edge=edge)
        if not np.isfinite(room).all() or room.min() < .10:
            self.pulse_until = 0.
            return None
        if not self.attempted:
            self.pulse_until = now + PULSE_SECONDS
            self.attempted = True
        return START_SPEED, 0.

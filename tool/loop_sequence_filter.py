from collections import deque

# Loopback candidate filter based on sequential temporal consistency
class SequenceConsistencyChecker:
    def __init__(self, window: int = 8, min_len: int = 4,
                 sim_thresh: float = 0.85, step_max: int = 3,
                 require_progress: bool = True):
        self.hist = deque(maxlen=window)
        self.min_len = min_len
        self.sim_thresh = sim_thresh
        self.step_max = step_max
        self.require_progress = require_progress

    def update(self, curr_idx: int, cand_idx: int, sim: float):
        """Each keyframe is called after the candidate for the best appearance is calculated. Return (confirmed, run_len)。"""
        self.hist.append((curr_idx, cand_idx, sim))
        return self._longest_run()

    def _longest_run(self):
        h = list(self.hist)
        if not h or h[-1][2] < self.sim_thresh:
            return False, 0
        run = [h[-1]]
        direction = 0
        moved = False

        for k in range(len(h) - 2, -1, -1):
            c = h[k]
            n = run[-1]
            if n[0] - c[0] != 1:
                break
            if c[2] < self.sim_thresh:
                break
            step = n[1] - c[1]
            if abs(step) > self.step_max:
                break
            if step != 0:
                d = 1 if step > 0 else -1
                if direction == 0:
                    direction = d
                elif d != direction:
                    break
                moved = True
            run.append(c)
        confirmed = len(run) >= self.min_len
        if self.require_progress and not moved:
            confirmed = False
        return confirmed, len(run)

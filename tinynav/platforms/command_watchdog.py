import math
import threading


def stop_unitree_motion(client, robot_model: str):
    if robot_model == 'g1':
        return client.SetVelocity(0.0, 0.0, 0.0)
    return client.StopMove()


class VelocityCommandWatchdog:
    def __init__(self, timeout_s: float):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be finite and positive')
        self.timeout_s = float(timeout_s)
        self._last_nonzero_at = None
        self._armed = False
        self._generation = 0
        self._stopped_through = 0
        self._lock = threading.Lock()

    def observe_nonzero(self, now: float):
        with self._lock:
            self._generation += 1
            self._last_nonzero_at = float(now)
            self._armed = True
            return self._generation

    def request_stop(self) -> int:
        with self._lock:
            self._stopped_through = self._generation
            return self._generation

    def confirm_stop(self, generation: int):
        with self._lock:
            self._stopped_through = max(self._stopped_through, generation)
            if self._generation > generation:
                return
            self._last_nonzero_at = None
            self._armed = False

    def retry_after_stop_failure(self, now: float):
        with self._lock:
            self._last_nonzero_at = float(now)
            self._armed = True

    def completed_after_stop(self, generation: int) -> bool:
        with self._lock:
            return generation <= self._stopped_through

    def consume_expiration(self, now: float) -> bool:
        with self._lock:
            if not self._armed or self._last_nonzero_at is None:
                return False
            if float(now) - self._last_nonzero_at <= self.timeout_s:
                return False
            self._stopped_through = self._generation
            self._last_nonzero_at = None
            self._armed = False
            return True

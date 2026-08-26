import threading


class VelocityCommandWatchdog:
    def __init__(self, timeout_s: float):
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.timeout_s = float(timeout_s)
        self._last_nonzero_at = None
        self._armed = False
        self._lock = threading.Lock()

    def observe_nonzero(self, now: float):
        with self._lock:
            self._last_nonzero_at = float(now)
            self._armed = True

    def clear(self):
        with self._lock:
            self._last_nonzero_at = None
            self._armed = False

    def consume_expiration(self, now: float) -> bool:
        with self._lock:
            if not self._armed or self._last_nonzero_at is None:
                return False
            if float(now) - self._last_nonzero_at <= self.timeout_s:
                return False
            self._last_nonzero_at = None
            self._armed = False
            return True

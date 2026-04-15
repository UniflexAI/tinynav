from collections import deque
from dataclasses import dataclass


@dataclass
class TimestampEvent:
    kind: str
    timestamp: float
    payload: object


class TimestampEventSequencer:
    def __init__(self, max_queue_size=1000, release_delay=0.01):
        self.max_queue_size = max_queue_size
        self.release_delay = release_delay
        self.imu_queue = deque()
        self.stereo_queue = deque()

    def add_imu(self, imu_msg):
        return self._add_event(self.imu_queue, TimestampEvent(
            kind="imu",
            timestamp=self._stamp_to_seconds(imu_msg.header.stamp),
            payload=imu_msg,
        ))

    def add_stereo(self, left_msg, right_msg):
        return self._add_event(self.stereo_queue, TimestampEvent(
            kind="stereo",
            timestamp=self._stamp_to_seconds(left_msg.header.stamp),
            payload=(left_msg, right_msg),
        ))

    def flush_ready(self):
        ready = []
        while True:
            next_event = self._pop_next_ready()
            if next_event is None:
                break
            ready.append(next_event)
        return ready

    def _add_event(self, queue, event):
        queue.append(event)
        while len(queue) > self.max_queue_size:
            queue.popleft()
        return self.flush_ready()

    def _pop_next_ready(self):
        if self.imu_queue and self.stereo_queue:
            imu_event = self.imu_queue[0]
            stereo_event = self.stereo_queue[0]
            if imu_event.timestamp <= stereo_event.timestamp:
                return self.imu_queue.popleft()
            return self.stereo_queue.popleft()

        if self.imu_queue and self._is_safe_to_release(self.imu_queue[0], self.stereo_queue):
            return self.imu_queue.popleft()

        if self.stereo_queue and self._is_safe_to_release(self.stereo_queue[0], self.imu_queue):
            return self.stereo_queue.popleft()

        return None

    def _is_safe_to_release(self, event, other_queue):
        if not other_queue:
            return False
        return event.timestamp <= other_queue[0].timestamp + self.release_delay

    @staticmethod
    def _stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

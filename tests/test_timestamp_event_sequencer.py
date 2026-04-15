from types import SimpleNamespace

from tinynav.core.timestamp_event_sequencer import TimestampEventSequencer


class DummyMsg:
    def __init__(self, sec, nanosec=0):
        self.header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def test_timestamp_event_sequencer_orders_imu_and_stereo_by_timestamp():
    sequencer = TimestampEventSequencer(release_delay=0.01)

    ready = sequencer.add_stereo(DummyMsg(1, 20_000_000), DummyMsg(1, 21_000_000))
    assert ready == []

    ready = sequencer.add_imu(DummyMsg(1, 10_000_000))
    assert [event.kind for event in ready] == ["imu", "stereo"]


def test_timestamp_event_sequencer_releases_when_other_stream_catches_up():
    sequencer = TimestampEventSequencer(release_delay=0.01)

    assert sequencer.add_imu(DummyMsg(1, 0)) == []
    ready = sequencer.add_stereo(DummyMsg(1, 5_000_000), DummyMsg(1, 6_000_000))

    assert [event.kind for event in ready] == ["imu", "stereo"]

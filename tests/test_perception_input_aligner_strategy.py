from tinynav.core.perception_node import stamp2second


class DummyHeader:
    def __init__(self, sec, nanosec=0):
        self.stamp = type('Stamp', (), {'sec': sec, 'nanosec': nanosec})()


class DummyImu:
    def __init__(self, t):
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        self.header = type('Header', (), {'stamp': type('Stamp', (), {'sec': sec, 'nanosec': nanosec})()})()


class DummyImage:
    def __init__(self, t):
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        self.header = type('Header', (), {'stamp': type('Stamp', (), {'sec': sec, 'nanosec': nanosec})()})()


def test_stamp2second_smoke():
    msg = DummyImu(0.123)
    assert abs(stamp2second(msg.header.stamp) - 0.123) < 1e-6

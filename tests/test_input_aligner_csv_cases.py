import os

from builtin_interfaces.msg import Time as TimeMsg
from message_filters import InputAligner, SimpleFilter
from rclpy.duration import Duration
from rclpy.time import Time


class Header:
    def __init__(self, stamp=None):
        self.stamp = stamp if stamp is not None else TimeMsg()


class ImuMsg:
    def __init__(self, stamp=None):
        self.header = Header(stamp)


class StereoMsg:
    def __init__(self, stamp=None):
        self.header = Header(stamp)


BUFFER_T = 0.055
DEBUG_INPUT_ALIGNER_CASES = os.environ.get('DEBUG_INPUT_ALIGNER_CASES', '').lower() in {'1', 'true', 'yes', 'on'}


IDEAL_CASE = {
    'inputs': [
        ('imu', 0.000),
        ('stereo', 0.000),
        ('imu', 0.010),
        ('imu', 0.020),
        ('imu', 0.030),
        ('imu', 0.040),
        ('imu', 0.050),
        ('imu', 0.060),
        ('imu', 0.070),
        ('imu', 0.080),
        ('imu', 0.090),
        ('imu', 0.100),
        ('stereo', 0.100),
        ('imu', 0.110),
        ('imu', 0.120),
        ('imu', 0.130),
        ('imu', 0.140),
        ('imu', 0.150),
        ('imu', 0.160),
        ('imu', 0.170),
    ],
    'expected': [
        ('imu', 0.000),
        ('stereo', 0.000),
        ('imu', 0.010),
        ('imu', 0.020),
        ('imu', 0.030),
        ('imu', 0.040),
        ('imu', 0.050),
        ('imu', 0.060),
        ('imu', 0.070),
        ('imu', 0.080),
        ('imu', 0.090),
        ('imu', 0.100),
        ('stereo', 0.100),
        ('imu', 0.110),
    ],
}

NORMAL_CASE = {
    'inputs': [
        ('imu', 0.000),
        ('imu', 0.010),
        ('imu', 0.020),
        ('stereo', 0.000),
        ('imu', 0.030),
        ('imu', 0.040),
        ('imu', 0.050),
        ('imu', 0.060),
        ('imu', 0.070),
        ('imu', 0.080),
        ('imu', 0.090),
        ('imu', 0.100),
        ('imu', 0.110),
        ('imu', 0.120),
        ('stereo', 0.100),
        ('imu', 0.130),
        ('imu', 0.140),
        ('imu', 0.150),
        ('imu', 0.160),
        ('imu', 0.170),
        ('imu', 0.180),
        ('imu', 0.190),
        ('imu', 0.200),
    ],
    'expected': [
        ('imu', 0.000),
        ('stereo', 0.000),
        ('imu', 0.010),
        ('imu', 0.020),
        ('imu', 0.030),
        ('imu', 0.040),
        ('imu', 0.050),
        ('imu', 0.060),
        ('imu', 0.070),
        ('imu', 0.080),
        ('imu', 0.090),
        ('imu', 0.100),
        ('stereo', 0.100),
        ('imu', 0.110),
        ('imu', 0.120),
        ('imu', 0.130),
        ('imu', 0.140),
    ],
}

IMU_DELAY_CASE = {
    'inputs': [
        ('imu', 0.000),
        ('imu', 0.010),
        ('imu', 0.020),
        ('stereo', 0.000),
        ('imu', 0.030),
        ('imu', 0.040),
        ('imu', 0.050),
        ('imu', 0.060),
        ('imu', 0.070),
        ('imu', 0.080),
        ('stereo', 0.100),
        ('imu', 0.090),
        ('imu', 0.100),
        ('imu', 0.110),
        ('imu', 0.120),
        ('imu', 0.130),
        ('imu', 0.140),
        ('imu', 0.150),
        ('imu', 0.160),
        ('imu', 0.170),
        ('imu', 0.180),
        ('imu', 0.190),
        ('imu', 0.200),
        ('imu', 0.210),
        ('imu', 0.220),
        ('stereo', 0.200),
        ('imu', 0.230),
        ('imu', 0.240),
        ('imu', 0.250),
        ('imu', 0.260),
        ('imu', 0.270),
    ],
    'expected': [
        ('imu', 0.000),
        ('stereo', 0.000),
        ('imu', 0.010),
        ('imu', 0.020),
        ('imu', 0.030),
        ('imu', 0.040),
        ('imu', 0.050),
        ('imu', 0.060),
        ('imu', 0.070),
        ('imu', 0.080),
        ('imu', 0.090),
        ('imu', 0.100),
        ('stereo', 0.100),
        ('imu', 0.110),
        ('imu', 0.120),
        ('imu', 0.130),
        ('imu', 0.140),
        ('imu', 0.150),
        ('imu', 0.160),
        ('imu', 0.170),
        ('imu', 0.180),
        ('imu', 0.190),
        ('imu', 0.200),
        ('stereo', 0.200),
        ('imu', 0.210),
    ],
}


def _build_msg(kind, t_sec):
    stamp = Time(nanoseconds=int(t_sec * 1e9)).to_msg()
    if kind == 'imu':
        return ImuMsg(stamp=stamp)
    if kind == 'stereo':
        return StereoMsg(stamp=stamp)
    raise ValueError(kind)


def _run_case(case, debug=DEBUG_INPUT_ALIGNER_CASES):
    imu_filter = SimpleFilter()
    stereo_filter = SimpleFilter()
    aligner = InputAligner(Duration(seconds=BUFFER_T), imu_filter, stereo_filter)
    aligner.setInputPeriod(0, Duration(seconds=0.01))
    aligner.setInputPeriod(1, Duration(seconds=0.1))

    actual_outputs = []

    def on_imu(msg):
        stamp = round(Time.from_msg(msg.header.stamp).nanoseconds / 1e9, 3)
        actual_outputs.append(('imu', stamp))
        if debug:
            print(f'callback imu {stamp:.3f}')

    def on_stereo(msg):
        stamp = round(Time.from_msg(msg.header.stamp).nanoseconds / 1e9, 3)
        actual_outputs.append(('stereo', stamp))
        if debug:
            print(f'callback stereo {stamp:.3f}')

    aligner.registerCallback(0, on_imu)
    aligner.registerCallback(1, on_stereo)

    for input_type, input_t in case['inputs']:
        msg = _build_msg(input_type, input_t)
        queue_index = 0 if input_type == 'imu' else 1
        if debug:
            print(f'add {input_type} {input_t:.3f} -> queue {queue_index}')
        aligner.add(msg, queue_index)
        if debug:
            q0 = [round(stamp.nanoseconds / 1e9, 3) for stamp, _ in aligner.event_queues[0].events]
            q1 = [round(stamp.nanoseconds / 1e9, 3) for stamp, _ in aligner.event_queues[1].events]
            print(f'  queued q0={q0} q1={q1}')
        if input_type == 'stereo':
            if debug:
                print('  dispatch on stereo boundary')
            aligner.dispatchMessages()
            if debug:
                q0 = [round(stamp.nanoseconds / 1e9, 3) for stamp, _ in aligner.event_queues[0].events]
                q1 = [round(stamp.nanoseconds / 1e9, 3) for stamp, _ in aligner.event_queues[1].events]
                print(f'  after dispatch q0={q0} q1={q1}')

    max_expected_t = max((t for _, t in case['expected']), default=0.0)
    flush_msg = _build_msg('imu', max_expected_t + 1.0)
    if debug:
        print(f'flush imu {max_expected_t + 1.0:.3f}')
    aligner.add(flush_msg, 0)
    aligner.dispatchMessages()

    return actual_outputs


def test_ideal_case_matches_input_aligner_replay():
    actual = _run_case(IDEAL_CASE)
    expected = IDEAL_CASE['expected']
    assert actual[:len(expected)] == expected


def test_normal_case_matches_input_aligner_replay():
    actual = _run_case(NORMAL_CASE)
    expected = NORMAL_CASE['expected']
    assert actual[:len(expected)] == expected


def test_imu_delay_case_matches_input_aligner_replay():
    actual = _run_case(IMU_DELAY_CASE)
    expected = IMU_DELAY_CASE['expected']
    assert actual[:len(expected)] == expected


if __name__ == '__main__':
    for name, case in [('ideal', IDEAL_CASE), ('normal', NORMAL_CASE), ('imu_delay', IMU_DELAY_CASE)]:
        if DEBUG_INPUT_ALIGNER_CASES:
            print(name)
            print(f'expected: {case["expected"]}')
        actual = _run_case(case)
        if DEBUG_INPUT_ALIGNER_CASES:
            print(name)
            print(f'acual: {actual}')
        assert actual[:len(case['expected'])] == case['expected']
    print('All input aligner timing cases passed.')

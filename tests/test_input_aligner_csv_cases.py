import csv
from pathlib import Path

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


def _load_cases():
    path = Path(__file__).parent / 'data' / 'timestamp_event_sequencer_cases.csv'
    sections = {}
    current = None
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            if row[0] == 'CASE' and len(row) > 1 and row[1] == 'seq':
                continue
            case = row[0].strip()
            if case and case != 'CASE':
                current = case
                sections.setdefault(case, []).append(row)
    return sections


def _collect_expected_outputs(rows):
    outputs = []
    for row in rows:
        if len(row) >= 7 and row[5].strip() and row[6].strip():
            outputs.append((row[5].strip(), float(row[6].strip())))
    return outputs


def _build_msg(kind, t_sec):
    stamp = Time(nanoseconds=int(t_sec * 1e9)).to_msg()
    if kind == 'imu':
        return ImuMsg(stamp=stamp)
    if kind == 'stereo':
        return StereoMsg(stamp=stamp)
    raise ValueError(kind)


def _run_case(rows):
    imu_filter = SimpleFilter()
    stereo_filter = SimpleFilter()
    aligner = InputAligner(Duration(seconds=BUFFER_T), imu_filter, stereo_filter)
    aligner.setInputPeriod(0, Duration(seconds=0.01))
    aligner.setInputPeriod(1, Duration(seconds=0.1))

    actual_outputs = []

    def on_imu(msg):
        actual_outputs.append(('imu', round(Time.from_msg(msg.header.stamp).nanoseconds / 1e9, 3)))

    def on_stereo(msg):
        actual_outputs.append(('stereo', round(Time.from_msg(msg.header.stamp).nanoseconds / 1e9, 3)))

    aligner.registerCallback(0, on_imu)
    aligner.registerCallback(1, on_stereo)

    for row in rows:
        if len(row) < 4:
            continue
        input_type = row[2].strip() if len(row) > 2 else ''
        input_t = row[3].strip() if len(row) > 3 else ''
        if input_type and input_t:
            msg = _build_msg(input_type, float(input_t))
            if input_type == 'imu':
                aligner.add(msg, 0)
            else:
                aligner.add(msg, 1)
            aligner.dispatchMessages()

    max_expected_t = max((t for _, t in _collect_expected_outputs(rows)), default=0.0)
    flush_msg = _build_msg('imu', max_expected_t + 1.0)
    aligner.add(flush_msg, 0)
    aligner.dispatchMessages()

    return actual_outputs


def test_csv_case_ideal_matches_input_aligner_replay():
    cases = _load_cases()
    expected = _collect_expected_outputs(cases['IDEAL'])
    actual = _run_case(cases['IDEAL'])
    assert actual[:len(expected)] == expected


def test_csv_case_normal_matches_input_aligner_replay():
    cases = _load_cases()
    expected = _collect_expected_outputs(cases['NORMAL'])
    actual = _run_case(cases['NORMAL'])
    assert actual[:len(expected)] == expected


def test_csv_case_imu_delay_matches_input_aligner_replay():
    cases = _load_cases()
    expected = _collect_expected_outputs(cases['IMU_DELAY'])
    actual = _run_case(cases['IMU_DELAY'])
    assert actual[:len(expected)] == expected

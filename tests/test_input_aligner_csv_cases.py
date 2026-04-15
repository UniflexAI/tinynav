import csv
from pathlib import Path


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
            if case and case not in {'CASE'}:
                current = case
                sections.setdefault(case, []).append(row)
        return sections


def _collect_expected_outputs(rows):
    outputs = []
    for row in rows:
        if len(row) >= 7 and row[5].strip() and row[6].strip():
            outputs.append((row[5].strip(), float(row[6].strip())))
    return outputs


def test_csv_case_ideal_has_expected_release_order_shape():
    cases = _load_cases()
    outputs = _collect_expected_outputs(cases['IDEAL'])
    assert outputs[0] == ('imu', 0.0)
    assert outputs[1] == ('stereo', 0.0)
    assert outputs[-1] == ('imu', 0.11)
    assert len(outputs) == 13


def test_csv_case_normal_has_expected_release_order_shape():
    cases = _load_cases()
    outputs = _collect_expected_outputs(cases['NORMAL'])
    assert outputs[0] == ('imu', 0.0)
    assert outputs[1] == ('stereo', 0.0)
    assert outputs[-1] == ('imu', 0.14)
    assert len(outputs) == 16


def test_csv_case_imu_delay_has_expected_release_order_shape():
    cases = _load_cases()
    outputs = _collect_expected_outputs(cases['IMU_DELAY'])
    assert outputs[0] == ('imu', 0.0)
    assert outputs[1] == ('stereo', 0.0)
    assert ('stereo', 0.1) in outputs
    assert ('stereo', 0.2) in outputs
    assert outputs[-1] == ('imu', 0.21)
    assert len(outputs) == 18

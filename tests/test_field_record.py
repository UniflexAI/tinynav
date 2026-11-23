"""Recorder lifecycle and bag verification, without constructing a robot node."""
import json
import signal
import subprocess
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from app.backend.node_manager import BackendNode


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    n = SimpleNamespace(_lock=threading.Lock(), tinynav_db_path=str(tmp_path),
        _field_record_proc=None, _field_record_path=None, _field_record_start=None,
        _field_record_stopping=False, _field_record_error=None, _field_record_messages=None,
        _field_record_verified=False, _field_record_topic_counts={}, _stair_mode='down', _stair_status='down ok', _sensor_mode='looper',
        FIELD_RECORD_TOPICS=BackendNode.FIELD_RECORD_TOPICS, FIELD_RECORD_INPUT_TOPICS=BackendNode.FIELD_RECORD_INPUT_TOPICS,
        stair_settings=lambda: dict(camera_height=.47, floors=1, landings_per_floor=2), get_logger=lambda: Mock())
    for name in ('cmd_field_record_start', 'cmd_field_record_stop', '_write_field_record_info', 'field_record_state', 'field_record_summary'):
        setattr(n, name, MethodType(getattr(BackendNode, name), n))
    proc = SimpleNamespace(pid=12345, returncode=None)
    proc.poll = lambda: proc.returncode
    proc.wait = Mock(side_effect=lambda **kw: setattr(proc, 'returncode', 0))
    def launch(name, cmd):
        from pathlib import Path
        Path(cmd[cmd.index('--output')+1]).mkdir(parents=True)
        return proc
    n._launch_proc = Mock(side_effect=launch)
    n._kill_proc = Mock(side_effect=lambda p: setattr(p, 'returncode', -9))
    monkeypatch.setattr('app.backend.node_manager.os.getpgid', lambda pid: pid)
    monkeypatch.setattr('app.backend.node_manager.os.killpg', Mock())
    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout='Messages: 42\n' if cmd[:3] == ['ros2','bag','info'] else 'revision-test\n')
    monkeypatch.setattr('app.backend.node_manager.subprocess.run', run)
    return n, proc


def test_recording_covers_raw_inputs_derived_outputs_and_optional_velocity_feedback():
    topics=BackendNode.FIELD_RECORD_TOPICS
    assert len(topics)==len(set(topics))
    assert set(('/cmd_vel','/slam/depth','/slam/odometry','/slam/odometry_visual','/planning/height_map',
                '/camera/camera/infra2/image_rect_raw','/camera/camera/infra2/camera_info','/sportmodestate','/tf','/tf_static')).issubset(topics)


def test_start_failure_is_reported_instead_of_recording_success(recorder):
    n,proc=recorder;proc.returncode=2
    with pytest.raises(RuntimeError,match='Recorder exited'):n.cmd_field_record_start()
    assert not n.field_record_summary()['recording']
    assert 'code 2' in n.field_record_summary()['error']


def test_stop_uses_sigint_checks_bag_and_preserves_start_context(recorder):
    from pathlib import Path
    from app.backend.node_manager import os
    n,proc=recorder;n.cmd_field_record_start()
    p=Path(n._field_record_path);initial=json.loads((p/'info.json').read_text())
    (p/'metadata.yaml').write_text('bag')
    n._stair_mode=None
    n.cmd_field_record_stop()
    os.killpg.assert_called_once_with(12345,signal.SIGINT)
    summary=n.field_record_summary()
    assert not summary['recording'] and summary['verified'] and summary['message_count']==42
    info=json.loads((p/'info.json').read_text())
    assert info['started_at_unix']==initial['started_at_unix']
    assert info['saved_stair_settings_at_start']==dict(camera_height=.47,floors=1,landings_per_floor=2)
    assert info['verified'] and info['error'] is None


@pytest.mark.parametrize('kind',['empty','missing','timeout'])
def test_empty_missing_or_forced_closed_bag_is_not_marked_verified(recorder,monkeypatch,kind):
    from pathlib import Path
    n,proc=recorder;n.cmd_field_record_start();p=Path(n._field_record_path)
    if kind!='missing':(p/'metadata.yaml').write_text('bag')
    if kind=='empty':monkeypatch.setattr('app.backend.node_manager.subprocess.run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='Messages: 0\n'))
    if kind=='timeout':proc.wait.side_effect=subprocess.TimeoutExpired('ros2',10)
    n.cmd_field_record_stop()
    summary=n.field_record_summary()
    assert not summary['verified'] and summary['error']
    assert not summary['recording'] and not summary['saving']


def test_start_cannot_replace_a_recording_while_it_is_being_saved(recorder):
    n,_=recorder;n._field_record_stopping=True
    with pytest.raises(RuntimeError,match='still being saved'):n.cmd_field_record_start()
    n._launch_proc.assert_not_called()


def test_api_returns_recorder_failure_as_503(recorder,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.backend.routers import field_record
    n,proc=recorder;proc.returncode=3
    monkeypatch.setattr(field_record.runner,'node',n)
    app=FastAPI();app.include_router(field_record.router,prefix='/field-record')
    client=TestClient(app)
    assert client.post('/field-record/start').status_code==503
    assert not client.get('/field-record/info').json()['recording']


def test_valid_bag_reports_missing_inputs_without_claiming_sensor_completeness(recorder):
    from pathlib import Path
    n,_=recorder;n.cmd_field_record_start();Path(n._field_record_path,'metadata.yaml').write_text('bag')
    n.cmd_field_record_stop()
    summary=n.field_record_summary()
    assert summary['verified'] # file integrity, not sensor completeness
    assert '/slam/depth' in summary['missing_topics']

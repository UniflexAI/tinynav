import threading
import time

import pytest

from tool.stair_sim_api import SimulationControl, validate_config


@pytest.mark.parametrize('values', [dict(floors=0), dict(floors=50), dict(floors=True),
    dict(direction='left'), dict(noise_scale=4), dict(speed=0), dict(unknown=1)])
def test_rejects_out_of_course_or_unknown_settings(values):
    with pytest.raises(ValueError):
        validate_config(values)


def test_command_acknowledges_only_after_loop_execution():
    control = SimulationControl()
    result = []
    thread = threading.Thread(target=lambda: result.append(control.submit('start', {'direction': 'up'}, timeout=2)))
    thread.start()
    command = control.commands.get(timeout=1)
    assert not result
    assert command.payload['floors'] == 9
    control.publish({'run_id': 'new-up', 'paused': False})
    command.result = control.snapshot()
    command.done.set()
    thread.join(1)
    assert result == [{'run_id': 'new-up', 'paused': False}]


def test_expired_command_never_executes_after_timeout():
    control = SimulationControl()
    with pytest.raises(TimeoutError):
        control.submit('reset', {}, timeout=.001)
    assert control.next_command() is None


def test_state_snapshot_is_not_mutable_by_http_consumer():
    control = SimulationControl()
    control.publish({'ready': True})
    control.snapshot()['ready'] = False
    assert control.snapshot()['ready']


def test_app_profile_has_no_robot_lifecycle_and_forwards_ack(monkeypatch):
    from fastapi.testclient import TestClient
    from app.backend import simulation_app
    calls = []
    def fake_call(path, payload=None, **kwargs):
        calls.append((path, payload))
        return {'ready': True, 'run_id': 'up-9', 'paused': False}
    monkeypatch.setattr(simulation_app, 'call', fake_call)
    client = TestClient(simulation_app.app)
    response = client.post('/simulation/start', json={'direction': 'up'})
    assert response.status_code == 200
    assert response.json()['run_id'] == 'up-9'
    assert calls[0][0] == '/start' and calls[0][1]['floors'] == 9
    assert client.post('/simulation/start', json={'floors': 50}).status_code == 422
    assert len(calls) == 1
    assert client.post('/action/command', json={'command': 'stand'}).status_code == 404
    assert 'simulation' in client.get('/device/info').json()['capabilities']


def test_original_app_stairs_uses_applied_noise_and_speed(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.backend import simulation_app as module
    calls = []
    def fake_call(path, payload=None, **kwargs):
        calls.append((path, payload))
        if path == '/state':
            return {'ready': True, 'floors': 9, 'direction': 'down', 'speed': 4,
                    'landing_depth': 1.2, 'odom_noise': {'mode': 'combined', 'scale': 0.5, 'seed': 7}}
        return {'run_id': 'app-up-1'}
    monkeypatch.setattr(module, 'call', fake_call)
    monkeypatch.setattr(module, '_SETTINGS_PATH', tmp_path/'settings.json')
    client = TestClient(module.app)
    result = client.post('/nav/stair/start', json={'direction': 'up', 'turn': 'auto', 'floors': 1, 'landings_per_floor': 2})
    assert result.status_code == 200 and result.json()['run_id'] == 'app-up-1'
    assert calls[-1] == ('/start', {'direction': 'up', 'floors': 1, 'landing_depth': 1.2,
                                   'speed': 4, 'noise_mode': 'combined', 'noise_scale': 0.5, 'noise_seed': 7})
    assert client.post('/nav/stair/start', json={'direction': 'up', 'floors': 50}).status_code == 422
    assert client.post('/nav/stair/start', json={'direction': 'up', 'turn': 'left'}).status_code == 422
    assert client.get('/sensor/image-topics').json()['topics'] == ['/camera/camera/color/image_raw', '/slam/depth']
    assert client.get('/map/current').status_code == 404
    assert client.get('/map/pois').json() == {'pois': []}
    assert client.put('/nav/stair/settings', json={'floors': 2, 'landings_per_floor': 2, 'camera_height': 0.47}).status_code == 200
    assert client.get('/nav/stair/settings').json()['floors'] == 2


def test_original_app_websockets_send_structured_planning_and_jpeg(monkeypatch):
    from fastapi.testclient import TestClient
    from app.backend import simulation_app as module
    requested = []
    snapshot = {'localized': False, 'odom_pose': {'x': 1., 'y': 2., 'z': 3., 'yaw': 0.1},
                'trajectory': [{'x': 1., 'y': 2.}], 'voxel_points': [{'x': 1., 'y': 2., 'z': 3.}]}
    jpeg = b'\xff\xd8test-jpeg\xff\xd9'
    def fake_call(path, payload=None, **kwargs):
        requested.append(path)
        return jpeg if path.startswith('/images/') else snapshot
    monkeypatch.setattr(module, 'call', fake_call)
    client = TestClient(module.app)
    with client.websocket_connect('/ws/planning') as ws:
        assert ws.receive_json() == snapshot
    with client.websocket_connect('/ws/pose') as ws:
        assert ws.receive_json() == snapshot['odom_pose']
    with client.websocket_connect('/ws/preview?topic=/camera/camera/color/image_raw&quality=high') as ws:
        assert ws.receive_bytes() == jpeg
    with client.websocket_connect('/ws/preview?topic=/slam/depth&quality=default') as ws:
        assert ws.receive_bytes() == jpeg
    assert '/images/color-high' in requested and '/images/depth' in requested


@pytest.mark.parametrize('command', [dict(linear_x=float('nan')), dict(linear_y=.51),
    dict(angular_z=float('inf')), dict(angular_z=-1.01)])
def test_teleop_rejects_nonfinite_or_out_of_range_commands_before_queueing(command):
    control=SimulationControl()
    with pytest.raises(ValueError):control.submit('teleop',command,timeout=.01)
    assert control.commands.empty()


def test_original_app_teleop_forwards_motion_and_zeros_on_disconnect(monkeypatch):
    from fastapi.testclient import TestClient
    from app.backend import simulation_app as module
    calls=[];forwarded=threading.Event();stopped=threading.Event()
    def fake_call(path,payload=None,**kwargs):
        calls.append((path,payload))
        if payload.get('linear_x'):forwarded.set()
        else:stopped.set()
        return {'ok':True}
    monkeypatch.setattr(module,'call',fake_call)
    client=TestClient(module.app)
    with client.websocket_connect('/ws/teleop') as ws:
        ws.send_json(dict(linear_x=.2,linear_y=0.,angular_z=.1))
        assert forwarded.wait(1.)
    assert calls[0]==('/teleop',dict(linear_x=.2,linear_y=0.,angular_z=.1))
    assert stopped.wait(1.)
    assert calls[-1]==('/teleop',dict(linear_x=0.,linear_y=0.,angular_z=0.))

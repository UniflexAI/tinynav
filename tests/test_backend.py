"""
Backend tests — runnable as a plain Python script following the project convention.

Usage:
    cd /tinynav
    python tests/test_backend.py          # layers 1 & 2 (no fastapi required)
    python tests/test_backend.py --routes  # include router tests (requires fastapi + httpx)
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------------------
# Simple test runner
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _run(name: str, fn):
    try:
        fn()
        _results.append((name, True, ''))
        print(f'  PASS  {name}')
    except Exception as e:
        _results.append((name, False, str(e)))
        print(f'  FAIL  {name}')
        traceback.print_exc()


def _summary():
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f'\n{passed}/{total} tests passed.')
    return passed == total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_map(tmp: str, grid: np.ndarray, meta: np.ndarray):
    np.save(os.path.join(tmp, 'occupancy_grid.npy'), grid)
    np.save(os.path.join(tmp, 'occupancy_meta.npy'), meta)


# ---------------------------------------------------------------------------
# Layer 1 — map_renderer (pure numpy / PIL, no ROS)
# ---------------------------------------------------------------------------

from app.backend.map_renderer import render_map


def test_render_returns_valid_png():
    grid = np.zeros((10, 10, 5), dtype=np.uint8)
    meta = np.array([0.0, 0.0, 0.0, 0.05])
    with tempfile.TemporaryDirectory() as tmp:
        _write_map(tmp, grid, meta)
        png, info = render_map(tmp)
    assert png[:4] == b'\x89PNG', 'Output must be a valid PNG'
    assert info['width'] == 10
    assert info['height'] == 10
    assert abs(info['resolution'] - 0.05) < 1e-9


def test_render_pixel_colours():
    from PIL import Image
    grid = np.zeros((4, 4, 2), dtype=np.uint8)
    grid[0, 0, 0] = 1   # free at world (X=0, Y=0)
    grid[1, 1, 0] = 2   # occupied at world (X=1, Y=1)
    meta = np.array([0.0, 0.0, 0.0, 0.1])
    with tempfile.TemporaryDirectory() as tmp:
        _write_map(tmp, grid, meta)
        png, _ = render_map(tmp)
    img = np.array(Image.open(io.BytesIO(png)).convert('RGB'))
    # After flipud+transpose: image row = (height-1-worldY), col = worldX
    free_pixel     = img[3, 0]   # world (X=0, Y=0) → row 3
    occupied_pixel = img[2, 1]   # world (X=1, Y=1) → row 2
    unknown_pixel  = img[1, 2]   # world (X=2, Y=2) → unknown
    assert free_pixel[0] > 150,      f'Free cell should be light, got {free_pixel}'
    assert occupied_pixel[0] < 100,  f'Occupied cell should be dark, got {occupied_pixel}'
    assert 60 < unknown_pixel[0] < 180, f'Unknown cell should be gray, got {unknown_pixel}'


def test_render_missing_files_raises():
    with tempfile.TemporaryDirectory() as tmp:
        raised = False
        try:
            render_map(tmp)
        except FileNotFoundError:
            raised = True
    assert raised, 'FileNotFoundError expected when files are absent'


def test_render_metadata_origin():
    grid = np.zeros((6, 8, 3), dtype=np.uint8)
    meta = np.array([-1.5, 2.0, 0.0, 0.2])
    with tempfile.TemporaryDirectory() as tmp:
        _write_map(tmp, grid, meta)
        _, info = render_map(tmp)
    assert abs(info['origin_x'] - (-1.5)) < 1e-9
    assert abs(info['origin_y'] - 2.0) < 1e-9
    assert info['width'] == 8
    assert info['height'] == 6


# ---------------------------------------------------------------------------
# Layer 2 — BackendNode static helpers (no rclpy.init() needed)
# ---------------------------------------------------------------------------

from app.backend.node_manager import BackendNode


def test_map_status_building():
    assert BackendNode._derive_map_status('rosbag_build_map', 50.0, False) == 'building'


def test_map_status_error():
    assert BackendNode._derive_map_status('error:build_map', 0.0, False) == 'failed'
    assert BackendNode._derive_map_status('error:perception', 0.0, True) == 'failed'


def test_map_status_success():
    assert BackendNode._derive_map_status('idle', 0.0, True) == 'success'


def test_map_status_idle():
    assert BackendNode._derive_map_status('idle', 0.0, False) == 'idle'


def test_map_status_building_overrides_files():
    assert BackendNode._derive_map_status('rosbag_build_map', 80.0, True) == 'building'


def _make_fake_odom(x, y, z, qx, qy, qz, qw, sec=1, nsec=0):
    class _Odom:
        class header:
            class stamp:
                pass
        class pose:
            class pose:
                class position:
                    pass
                class orientation:
                    pass
    o = _Odom()
    o.header.stamp.sec = sec
    o.header.stamp.nanosec = nsec
    o.pose.pose.position.x = x
    o.pose.pose.position.y = y
    o.pose.pose.position.z = z
    o.pose.pose.orientation.x = qx
    o.pose.pose.orientation.y = qy
    o.pose.pose.orientation.z = qz
    o.pose.pose.orientation.w = qw
    return o


def test_odom_position():
    d = BackendNode._odom_to_dict(_make_fake_odom(1, 2, 3, 0, 0, 0, 1), 'slam')
    assert abs(d['x'] - 1) < 1e-9
    assert abs(d['y'] - 2) < 1e-9
    assert abs(d['z'] - 3) < 1e-9


def test_odom_yaw_identity():
    d = BackendNode._odom_to_dict(_make_fake_odom(0, 0, 0, 0, 0, 0, 1), 'slam')
    assert abs(d['yaw']) < 1e-9


def test_odom_yaw_90():
    s = math.sin(math.pi / 4)
    d = BackendNode._odom_to_dict(_make_fake_odom(0, 0, 0, 0, 0, s, s), 'slam')
    assert abs(d['yaw'] - math.pi / 2) < 1e-6, f"Expected π/2, got {d['yaw']}"


def test_odom_source_field():
    d1 = BackendNode._odom_to_dict(_make_fake_odom(0, 0, 0, 0, 0, 0, 1), 'slam')
    d2 = BackendNode._odom_to_dict(_make_fake_odom(0, 0, 0, 0, 0, 0, 1), 'map')
    assert d1['source'] == 'slam'
    assert d2['source'] == 'map'


def test_odom_timestamp():
    d = BackendNode._odom_to_dict(_make_fake_odom(0, 0, 0, 0, 0, 0, 1, sec=5, nsec=500_000_000), 'slam')
    assert abs(d['timestamp'] - 5.5) < 1e-9


# ---------------------------------------------------------------------------
# Layer 3 — Router tests (requires fastapi + httpx)
# ---------------------------------------------------------------------------

def _run_router_tests():
    try:
        import fastapi  # noqa: F401
        from fastapi.testclient import TestClient
    except ImportError:
        print('\n  SKIP  router tests (fastapi not installed)')
        return

    from app.backend import state as app_state
    from app.backend.main import app

    class FakeNode:
        def __init__(self, map_path, bag_path):
            self.state = 'idle'
            self.map_path = map_path
            self.bag_path = bag_path
            self._started: list = []
            self._stopped = False
            self._nav_target = None
            self.pose_callbacks: list = []

        def get_status(self):
            grid = os.path.join(self.map_path, 'occupancy_grid.npy')
            bag  = os.path.join(self.bag_path, 'bag_0.db3')
            return {
                'bagStatus': 'recording' if self.state == 'realsense_bag_record' else 'idle',
                'bagFileReady': os.path.exists(bag),
                'mapStatus': BackendNode._derive_map_status(self.state, 0.0, os.path.exists(grid)),
                'mappingPercent': 0.0,
                'navStatus': 'navigating' if self.state == 'navigation' else 'idle',
                'rawState': self.state,
            }

        def cmd_bag_start(self):
            self.state = 'realsense_bag_record'
            self._started.append('bag')

        def cmd_bag_stop(self):
            self.state = 'idle'
            self._stopped = True

        def cmd_map_build(self):
            self.state = 'rosbag_build_map'
            self._started.append('map')

        def cmd_nav_start(self, poi_id=None):
            self.state = 'navigation'
            self._nav_target = poi_id
            self._started.append('nav')

        def cmd_nav_cancel(self):
            self.state = 'idle'
            self._stopped = True

    with tempfile.TemporaryDirectory() as tmp:
        map_path = os.path.join(tmp, 'map')
        bag_path = os.path.join(tmp, 'bag')
        os.makedirs(map_path)
        os.makedirs(bag_path)
        node = FakeNode(map_path=map_path, bag_path=bag_path)

        original = app_state.runner.node
        app_state.runner.node = node
        # TestClient without context manager skips the lifespan (no rclpy.init)
        client = TestClient(app, raise_server_exceptions=True)

        try:
            _run_router_suite(client, node, map_path, bag_path)
        finally:
            app_state.runner.node = original


def _run_router_suite(client, node, map_path, bag_path):
    def reset():
        node.state = 'idle'
        node._started.clear()
        node._stopped = False
        node._nav_target = None
        # clear pois
        pois_file = os.path.join(map_path, 'pois.json')
        if os.path.exists(pois_file):
            os.remove(pois_file)

    # -- device --
    def test_device_info():
        r = client.get('/device/info')
        assert r.status_code == 200
        assert 'deviceId' in r.json()
        assert 'bag_record' in r.json()['capabilities']

    def test_device_status_online():
        r = client.get('/device/status')
        assert r.status_code == 200
        assert r.json()['online'] is True

    # -- bag --
    def test_bag_start_from_idle():
        reset()
        r = client.post('/bag/start')
        assert r.status_code == 200
        assert node.state == 'realsense_bag_record'

    def test_bag_start_already_recording():
        reset()
        node.state = 'realsense_bag_record'
        assert client.post('/bag/start').status_code == 409

    def test_bag_stop():
        reset()
        node.state = 'realsense_bag_record'
        r = client.post('/bag/stop')
        assert r.status_code == 200
        assert node._stopped

    def test_bag_stop_when_idle():
        reset()
        assert client.post('/bag/stop').status_code == 409

    def test_bag_status_recording():
        reset()
        node.state = 'realsense_bag_record'
        assert client.get('/bag/status').json()['status'] == 'recording'

    # -- map --
    def test_map_build_no_bag():
        reset()
        assert client.post('/map/build').status_code == 400

    def test_map_build_with_bag():
        reset()
        open(os.path.join(bag_path, 'bag_0.db3'), 'w').close()
        r = client.post('/map/build')
        assert r.status_code == 200
        assert node.state == 'rosbag_build_map'

    def test_map_current_no_map():
        reset()
        assert client.get('/map/current').status_code == 404

    def test_map_current_and_image():
        reset()
        grid = np.zeros((8, 8, 4), dtype=np.uint8)
        grid[2, 2, 1] = 2
        _write_map(map_path, grid, np.array([0.0, 0.0, 0.0, 0.1]))
        r = client.get('/map/current')
        assert r.status_code == 200
        assert r.json()['imageUrl'] == '/map/image'
        r2 = client.get('/map/image')
        assert r2.status_code == 200
        assert r2.headers['content-type'] == 'image/png'
        assert r2.content[:4] == b'\x89PNG'

    # -- POI --
    def test_poi_list_empty():
        reset()
        r = client.get('/map/pois')
        assert r.status_code == 200
        assert r.json()['pois'] == []

    def test_poi_create_and_list():
        reset()
        r = client.post('/map/pois', json={'name': 'Kitchen', 'position': [1.0, 2.0, 0.0]})
        assert r.status_code == 200
        assert r.json()['name'] == 'Kitchen'
        assert len(client.get('/map/pois').json()['pois']) == 1

    def test_poi_delete():
        reset()
        poi_id = client.post('/map/pois', json={'name': 'A', 'position': [0.0, 0.0, 0.0]}).json()['id']
        assert client.delete(f'/poi/{poi_id}').status_code == 200
        assert client.get('/map/pois').json()['pois'] == []

    def test_poi_delete_nonexistent():
        reset()
        assert client.delete('/poi/999').status_code == 404

    def test_poi_invalid_position():
        reset()
        assert client.post('/map/pois', json={'name': 'Bad', 'position': [1.0, 2.0]}).status_code == 400

    def test_poi_unique_ids():
        reset()
        id1 = client.post('/map/pois', json={'name': 'A', 'position': [0.0, 0.0, 0.0]}).json()['id']
        id2 = client.post('/map/pois', json={'name': 'B', 'position': [1.0, 0.0, 0.0]}).json()['id']
        assert id1 != id2

    # -- nav --
    def test_nav_go_to_poi():
        reset()
        r = client.post('/nav/go-to-poi', json={'poi_id': 2})
        assert r.status_code == 200
        assert node.state == 'navigation'
        assert node._nav_target == '2'

    def test_nav_go_already_navigating():
        reset()
        node.state = 'navigation'
        assert client.post('/nav/go-to-poi', json={'poi_id': 0}).status_code == 409

    def test_nav_cancel():
        reset()
        node.state = 'navigation'
        assert client.post('/nav/cancel').status_code == 200
        assert node._stopped

    def test_nav_cancel_when_idle():
        reset()
        assert client.post('/nav/cancel').status_code == 409

    def test_nav_status_idle():
        reset()
        assert client.get('/nav/status').json()['status'] == 'idle'

    def test_nav_status_navigating():
        reset()
        node.state = 'navigation'
        assert client.get('/nav/status').json()['status'] == 'navigating'

    route_tests = [
        test_device_info, test_device_status_online,
        test_bag_start_from_idle, test_bag_start_already_recording,
        test_bag_stop, test_bag_stop_when_idle, test_bag_status_recording,
        test_map_build_no_bag, test_map_build_with_bag,
        test_map_current_no_map, test_map_current_and_image,
        test_poi_list_empty, test_poi_create_and_list, test_poi_delete,
        test_poi_delete_nonexistent, test_poi_invalid_position, test_poi_unique_ids,
        test_nav_go_to_poi, test_nav_go_already_navigating,
        test_nav_cancel, test_nav_cancel_when_idle,
        test_nav_status_idle, test_nav_status_navigating,
    ]
    for fn in route_tests:
        _run(fn.__name__, fn)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    run_routes = '--routes' in sys.argv

    print('\n=== Layer 1: map_renderer ===')
    for fn in [
        test_render_returns_valid_png,
        test_render_pixel_colours,
        test_render_missing_files_raises,
        test_render_metadata_origin,
    ]:
        _run(fn.__name__, fn)

    print('\n=== Layer 2: node logic ===')
    for fn in [
        test_map_status_building,
        test_map_status_error,
        test_map_status_success,
        test_map_status_idle,
        test_map_status_building_overrides_files,
        test_odom_position,
        test_odom_yaw_identity,
        test_odom_yaw_90,
        test_odom_source_field,
        test_odom_timestamp,
    ]:
        _run(fn.__name__, fn)

    if run_routes:
        print('\n=== Layer 3: routers ===')
        _run_router_tests()

    ok = _summary()
    sys.exit(0 if ok else 1)

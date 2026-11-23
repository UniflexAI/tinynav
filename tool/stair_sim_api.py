"""Loop-owned simulator control. HTTP threads never touch physics or ROS nodes."""
import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace


DEFAULT_CONFIG = dict(direction='down', floors=9, landing_depth=1.2,
                      noise_mode='combined', noise_scale=0.5, noise_seed=7, speed=1)


def validate_config(values):
    unknown = set(values) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError('Unknown settings: ' + ', '.join(sorted(unknown)))
    result = {**DEFAULT_CONFIG, **values}
    allowed = dict(direction=('up', 'down'), floors=tuple(range(1,10)), landing_depth=(1.2, 1.75),
                   noise_mode=('none', 'position', 'yaw', 'pitch', 'combined'),
                   noise_scale=(0.5, 1, 2), noise_seed=(7, 17, 27), speed=(1, 2, 4))
    for key in ('floors', 'noise_seed', 'speed'):
        if type(result[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    for key, options in allowed.items():
        if isinstance(result[key], bool) or result[key] not in options:
            raise ValueError(f'Invalid {key}: expected one of {options}')
    return result


class SimulationControl:
    def __init__(self):
        self.commands = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.state = {'ready': False}
        self.images = {}
        self.planning = {}

    def publish(self, state):
        with self.lock:
            self.state = dict(state)

    def image(self, name, data):
        with self.lock:
            self.images[name] = bytes(data)

    def publish_planning(self, snapshot):
        with self.lock:
            self.planning = snapshot

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def submit(self, action, payload, timeout=55):
        if action not in ('start', 'reset', 'pause', 'resume', 'stop', 'camera', 'teleop', 'vio_confirm', 'fault'):
            raise ValueError('Unknown action')
        if action in ('start', 'reset'):
            payload = validate_config(payload)
        elif action == 'camera':
            if payload.get('mode') not in ('Free orbit', 'Follow robot', 'Overview'):
                raise ValueError('Invalid camera mode')
        elif action == 'vio_confirm':
            if type(payload.get('fault_id')) is not int or payload['fault_id'] < 1:
                raise ValueError('Invalid fault ID')
        elif action == 'fault':
            if payload.get('kind') not in ('jump', 'dropout', 'restart'):
                raise ValueError('Invalid VIO fault')
            duration = payload.get('duration', 2.)
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not .5 <= duration <= 10.:
                raise ValueError('Invalid fault duration')
        if action == 'teleop':
            for key,limit in (('linear_x',.5),('linear_y',.5),('angular_z',1.)):
                value=float(payload.get(key,0.))
                if not math.isfinite(value) or abs(value)>limit:
                    raise ValueError(f'Invalid {key}')
                payload[key]=value
        command = SimpleNamespace(action=action, payload=payload, done=threading.Event(),
                                  deadline=time.monotonic()+timeout, error=None, result=None)
        self.commands.put_nowait(command)
        if not command.done.wait(timeout):
            raise TimeoutError('Simulator command timed out; inspect current state before retrying')
        if command.error:
            raise ValueError(command.error)
        return command.result

    def next_command(self):
        try:
            command = self.commands.get_nowait()
        except queue.Empty:
            return None
        if time.monotonic() > command.deadline:
            command.error = 'Expired command'
            command.done.set()
            return None
        return command

    def serve(self, port):
        control = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, data, kind='application/json'):
                body = json.dumps(data).encode() if kind == 'application/json' else data
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/state':
                    return self.reply(200, control.snapshot())
                if self.path == '/planning':
                    with control.lock:
                        snapshot = control.planning
                    return self.reply(200, snapshot)
                name = self.path.removeprefix('/images/')
                with control.lock:
                    frame = control.images.get(name)
                return self.reply(200, frame, 'image/jpeg') if frame else self.reply(404, {'detail': 'Frame not ready'})

            def do_POST(self):
                try:
                    size = int(self.headers.get('Content-Length', 0))
                    if not 0 <= size <= 8192:
                        raise ValueError('Request too large')
                    payload = json.loads(self.rfile.read(size) or b'{}')
                    if not isinstance(payload, dict):
                        raise ValueError('Expected a JSON object')
                    self.reply(200, control.submit(self.path.strip('/'), payload))
                except (ValueError, json.JSONDecodeError) as exc:
                    self.reply(409, {'detail': str(exc)})
                except (TimeoutError, queue.Full) as exc:
                    self.reply(503, {'detail': str(exc)})
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tool.simulator.decision_observer import DecisionObserver, QUESTIONS, validate_response


def good_response():
    return {"answers": {"action": {"choice": "wait", "probabilities": {k: float(k == "wait") for k in QUESTIONS["action"]["criteria"]}}, "stuck": {"noul": 0.0}}}


class ObserverTests(unittest.TestCase):
    def test_invalid_distribution(self):
        result = good_response()
        result["answers"]["action"]["probabilities"]["wait"] = float("nan")
        with self.assertRaises(ValueError):
            validate_response(result)

    def test_single_flight_snapshot_and_logging(self):
        entered, release = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                entered.set()
                release.wait(3)
                data = json.dumps(good_response()).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                observer = DecisionObserver(Path(folder), f"http://127.0.0.1:{server.server_port}/v1/systemone")
                state = {"goal": {"distance_m": 5}}
                observer.submit(state, {})
                self.assertTrue(entered.wait(2))
                state["goal"]["distance_m"] = 99
                with self.assertRaises(RuntimeError):
                    observer.submit(state, {})
                release.set()
                deadline = time.monotonic() + 3
                while observer.status()["status"] == "running" and time.monotonic() < deadline:
                    time.sleep(0.01)
                record = observer.status()
                self.assertEqual(record["status"], "complete")
                self.assertEqual(record["request"]["state"]["goal"]["distance_m"], 5)
                self.assertEqual(json.loads(next(Path(folder).glob("*.json")).read_text()), record)
        finally:
            release.set()
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

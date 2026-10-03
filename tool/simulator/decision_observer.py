"""One-shot decision observation. Never publishes navigation commands."""
import copy
import json
import math
import threading
import time
import uuid
from urllib import request, error

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Recommend a navigation planning action using only the supplied observations. Unknown space is not clear space. Directional rays are not robot footprint clearance. A paused robot is not stuck. Prefer wait when navigation is paused, and uncertain when evidence is insufficient. This is advice only.",
        "criteria": {
            "continue": "Current progress and measured clearance support continuing the planner",
            "wait": "A temporary obstacle or unavailable fresh observation warrants waiting",
            "suggest_left": "Measured observations support investigating the left side",
            "suggest_right": "Measured observations support investigating the right side",
            "replan": "Recent lack of progress or blockage warrants a new plan",
            "uncertain": "Observations do not support a reliable recommendation",
        },
    },
    "stuck": {"type": "noul", "instructions": "Do recent motion observations show the robot is stuck while navigation is running and still away from its goal? A paused robot is not stuck. Missing history is insufficient evidence."},
}


def validate_response(result):
    answers = result["answers"]
    action = answers["action"]
    probabilities = action["probabilities"]
    if set(probabilities) != set(QUESTIONS["action"]["criteria"]):
        raise ValueError("Unexpected action options")
    values = list(probabilities.values())
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Invalid action probabilities")
    if abs(sum(values) - 1) > 0.001 or action["choice"] not in probabilities:
        raise ValueError("Invalid action distribution")
    stuck = answers["stuck"]["noul"]
    if not isinstance(stuck, (int, float)) or not math.isfinite(stuck) or not 0 <= stuck <= 1:
        raise ValueError("Invalid stuck probability")


class DecisionObserver:
    def __init__(self, folder, endpoint="http://127.0.0.1:8090/v1/systemone"):
        self.folder = folder
        self.endpoint = endpoint
        self.lock = threading.Lock()
        self.current = None

    def status(self):
        with self.lock:
            return copy.deepcopy(self.current)

    def submit(self, world_state, context):
        with self.lock:
            if self.current and self.current["status"] == "running":
                raise RuntimeError("A decision request is already running")
            record = {"id": uuid.uuid4().hex, "status": "running", "created_at_unix": time.time(),
                      "context": copy.deepcopy(context), "request": {"state": copy.deepcopy(world_state), "questions": copy.deepcopy(QUESTIONS)},
                      "observer_only": True}
            self.current = record
        threading.Thread(target=self._run, args=(record,), daemon=True).start()
        return self.status()

    def _run(self, record):
        started = time.monotonic()
        outcome = {}
        try:
            req = request.Request(self.endpoint, data=json.dumps(record["request"]).encode(), headers={"Content-Type": "application/json"})
            with request.urlopen(req, timeout=30) as response:
                result = json.load(response)
            validate_response(result)
            outcome = {"status": "complete", "response": result}
        except (error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
            outcome = {"status": "error", "error": str(exc)}
        outcome["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        finished = {**record, **outcome}
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / (record["id"] + ".json")).write_text(json.dumps(finished, indent=2))
        except OSError as exc:
            finished["log_error"] = str(exc)
        with self.lock:
            self.current = finished

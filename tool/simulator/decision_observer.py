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


def validate_response(result, questions=None):
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

    for key, spec in (questions or {}).items():
        if spec["type"] != "choice":
            continue
        answer = answers[key]
        probs = answer["probabilities"]
        if set(probs) != set(spec["criteria"]) or answer["choice"] not in probs:
            raise ValueError("Unexpected choices for " + key)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > 0.001:
            raise ValueError("Invalid distribution for " + key)


def planning_questions(state):
    questions = copy.deepcopy(QUESTIONS)
    if "planning" not in state:
        return questions
    questions["planning_issue"] = {"type": "choice", "instructions": "Which issue best explains the current planning report and numerical motion history? Consider scores and target distance; do not infer safety from unknown observations.", "criteria": {
        "none": "Plan has plausible progress and no evident issue", "stalled_progress": "Numerical history shows little progress away from the goal",
        "collision_blocked": "Collision rejections eliminate useful candidates", "reverse_gate": "Reverse gating heavily penalizes otherwise useful candidates",
        "goal_sampling": "Velocity sampling or horizon prevents a useful near-goal step", "insufficient_observation": "Evidence is insufficient to diagnose"}}
    criteria = {"keep_current": "Keep the planner selection", "request_new_candidates": "No existing candidate provides a useful alternative", "uncertain": "Cannot establish a useful alternative from these observations"}
    for c in state["planning"]["top_candidates"]:
        if c.get("cost", 0) is not None and "sampled_footprint_collision" not in c["reasons"] and c["id"] != state["planning"]["selected_id"]:
            criteria["candidate_" + str(c["id"])] = "Consider this reported alternative, subject to planner collision checks: "
    questions["alternative"] = {"type": "choice", "instructions": "Recommend keeping the selected trajectory or investigating one reported alternative. Compare progress and direction diversity, not cost alone when motion is stalled. Reverse-gate penalties are policy penalties, not collision rejection. This is isolated analysis only; unknown space and sampled ESDF scores are not safety guarantees.", "criteria": criteria}
    if 'recovery' in state:
        criteria = {'keep_current':'Continue the original planner', 'request_new_candidates':'All available recovery sequences are unsuitable', 'uncertain':'Insufficient measured evidence to select a recovery sequence'}
        for plan in state['recovery']['strategies']:
            criteria['strategy_'+plan['id']] = 'Try the bounded stages described in recovery.strategies for '+plan['id']
        questions['alternative'] = {'type':'choice','instructions':'Choose a multi-stage recovery sequence for the stalled robot. Compare measured blockage, unknown_fraction, escape displacement, goal bearing and previous attempts. Retreat may temporarily increase goal distance to leave a local trap. Avoid repeating failed attempts. This is an isolated simulation experiment: each stage is independently checked and aborted if blocked. Unknown is not free; choose uncertain if observations do not support any sequence. Do not choose based only on immediate goal progress.', 'criteria':criteria}
    return questions


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
                      "context": copy.deepcopy(context), "request": {"state": copy.deepcopy(world_state), "questions": planning_questions(world_state)},
                      "observer_only": not context.get("closed_loop_experiment", False)}
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
            validate_response(result, record["request"]["questions"])
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

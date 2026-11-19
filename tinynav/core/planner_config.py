"""Planner selection shared by local planning, velocity control, and recording."""
import os
import platform
from pathlib import Path


def planner_mode(environ=None):
    environ = os.environ if environ is None else environ
    name = environ.get("TINYNAV_PLANNER", "library").strip().lower()
    modes = {"library": "library", "dwa": "library", "rl": "rl", "ppo": "rl"}
    if name not in modes:
        raise ValueError(f"Unsupported TINYNAV_PLANNER: {name!r}")
    return modes[name]


def shared_safety(environ=None):
    environ = os.environ if environ is None else environ
    default = "1" if environ.get("TINYNAV_PLANNER", "").strip().lower() in ("dwa", "ppo") else "0"
    return environ.get("TINYNAV_NAV_SHARED_SAFETY", default) == "1"


def route_guidance(environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("TINYNAV_PLANNER", "").strip().lower() == "ppo":
        return True
    return planner_mode(environ) == "rl" and environ.get("TINYNAV_RL_ROUTE_GUIDANCE") == "1"


def policy_engine_path(environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("TINYNAV_RL_ENGINE"):
        return environ["TINYNAV_RL_ENGINE"]
    if environ.get("TINYNAV_PLANNER", "").strip().lower() == "ppo":
        return str(Path(__file__).resolve().parents[1] / "models/rl_nav_policy_ppo_20261114.plan")
    return f"/tinynav/tinynav/models/rl_nav_policy_v7_{platform.machine()}.plan"

#!/usr/bin/env python3
"""Browser UI for the offline tinynav planning MVP."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tool.simulator.offline_planning_mvp import default_config, run, run_realtime_step_with_grid, run_rollout


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "offline_planning_web"
DEFAULT_OUTPUT_DIR = Path("/tinynav/outputs/offline_planning_mvp")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LARGE_SUMMARY_KEYS = {
    "selected_trajectory_xy",
    "selected_footprints_xy",
    "candidate_trajectories_xy",
    "closest_steps",
}


class RunRequest(BaseModel):
    config: dict[str, Any]
    steps: int | None = None
    advance_step: int | None = None
    reset: bool | None = None


app = FastAPI(title="TinyNav Offline Planning Lab")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="outputs")
REALTIME_STATE: dict[str, Any] = {"occupancy_grid": None, "grid_shape": None, "last_param": None}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/default-config")
def get_default_config() -> dict[str, Any]:
    config = default_config()
    config["output_dir"] = str(DEFAULT_OUTPUT_DIR)
    return config


@app.post("/api/run")
def run_scene(request: RunRequest) -> dict[str, Any]:
    config = request.config
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    config = json.loads(json.dumps(config))
    config.setdefault("name", "web_scene")
    config["output_dir"] = str(DEFAULT_OUTPUT_DIR)
    try:
        summary = run(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    png_path = Path(summary["output_png"])
    if not png_path.exists():
        raise HTTPException(status_code=500, detail=f"missing output image: {png_path}")
    png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return {
        "summary": summary,
        "image_data_url": f"data:image/png;base64,{png_b64}",
    }


@app.post("/api/rollout")
def rollout_scene(request: RunRequest) -> dict[str, Any]:
    config = request.config
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    config = json.loads(json.dumps(config))
    config.setdefault("name", "web_scene")
    config["output_dir"] = str(DEFAULT_OUTPUT_DIR)
    try:
        rollout = run_rollout(
            config,
            steps=request.steps or 30,
            advance_step=request.advance_step or 3,
            render_snapshots=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    diagnostic = rollout.get("diagnostic_summary") or {}
    for frame in rollout.get("frames", []):
        png_path = frame.get("diagnostic_png")
        if png_path:
            frame["diagnostic_image_url"] = f"/outputs/{Path(png_path).name}"
    if diagnostic:
        rollout["diagnostic_summary"] = {
            key: value for key, value in diagnostic.items() if key not in LARGE_SUMMARY_KEYS
        }
    return {
        "rollout": rollout,
        "summary": rollout.get("diagnostic_summary") or {},
    }


@app.post("/api/realtime-step")
def realtime_step(request: RunRequest) -> dict[str, Any]:
    config = request.config
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    config = json.loads(json.dumps(config))
    config.setdefault("name", "web_realtime")
    config["output_dir"] = str(DEFAULT_OUTPUT_DIR)
    try:
        grid_shape = tuple(int(v) for v in config["grid"]["shape"])
        if request.reset or REALTIME_STATE.get("grid_shape") != grid_shape:
            REALTIME_STATE["occupancy_grid"] = np.zeros(grid_shape, dtype=np.float64)
            REALTIME_STATE["grid_shape"] = grid_shape
            REALTIME_STATE["last_param"] = None
        if REALTIME_STATE.get("last_param") is not None:
            config.setdefault("planner", {})["last_param"] = REALTIME_STATE["last_param"]
        frame, occupancy_grid = run_realtime_step_with_grid(
            config,
            occupancy_grid_state=REALTIME_STATE.get("occupancy_grid"),
            advance_step=request.advance_step or 3,
        )
        REALTIME_STATE["occupancy_grid"] = occupancy_grid
        REALTIME_STATE["last_param"] = frame.get("selected_param")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"frame": frame}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8766)


if __name__ == "__main__":
    main()

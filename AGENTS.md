# Agents Guide
This repo hosts **TinyNav**, a ROS 2 Humble–based stereo SLAM + mapping + planning stack optimized for Jetson/desktop GPUs. Use this guide when automating edits or troubleshooting with an AI coding agent.

## Project Snapshot
- Core ROS 2 nodes: `perception_node.py` (stereo odom + depth + TensorRT feature/stereo models), `planning_node.py` (occupancy grid, ESDF, trajectory library scoring), `control_node.py`/`path_follow_node.py` (teleop + trajectory following), `driver_node.py` (KITTI playback), `map_node.py` (loop closure, pose graph, map merge, DINO features).
- Performance levers: Numba JIT in planners/mappers, C++ bindings in `tinynav/cpp` (`tinynav_cpp_bind` via scikit-build), TensorRT engines in `tinynav/models`.
- Typical ROS topics: `/camera/camera/infra1|infra2/image_rect_raw`, `/camera/camera/accel|gyro/sample`, `/slam/odometry`, `/slam/depth`|`/slam/disparity_vis`, `/slam/keyframe_*`, `/planning/trajectory_path`, `/planning/height_map`, `/cmd_vel`.

## Repo Map
- `tinynav/core/`: main ROS 2 nodes, math utils, TRT model wrappers.
- `tinynav/platforms/`: control bindings for LeKiwi (`lekiwi_control.py`) and simulator.
- `tinynav/cpp/`: raycasting, pose graph solve, bundle adjustment; built into `tinynav_cpp_bind`.
- `tinynav/models/`: prebuilt TensorRT engines (SuperPoint, LightGlue, stereo fusion, etc).
- `scripts/`: tmux launchers (`run_navigation.sh`, `run_planning.sh`, `run_rosbag_examples.sh`, `run_rosbag_build_map.sh`, `run_realsense_*`), env checks, 3DGS generation helper.
- `docs/`: quickstart docs and RViz config.
- `tinynav_db/`, `tinynav_temp/`, `tinynav_map/`: stored map artifacts, shelve DBs (`TinyNavDB` in `build_map_node.py`).
- `tests/`: unit/benchmark tests (`test_planning_node.py`, `test_map_node.py`, `test_func.py`, disjoint set tests, TRT loading checks).

## Environment & Setup
- Python 3.10.x, ROS 2 Humble. GPU strongly recommended; TRT engines expect NVIDIA runtime.
- Preferred flow: open in Dev Container (`devcontainer up --workspace-folder .`), then:
  ```bash
  uv venv --system-site-packages
  uv sync
  uv run pip install -e .  # builds C++ extension via scikit-build
  ```
- Quick sanity check (Docker, LFS, NVIDIA runtime patching): `bash scripts/check_env.sh`.
- Optional extras in `pyproject.toml`: `lekiwi`, `unitree`, `3dgs` (LightGlue from GitHub, nerfstudio, etc).

## Running Pipelines
- Demo with sample rosbag + RViz: `bash scripts/run_rosbag_examples.sh`.
- Planning-only tmux session: `bash scripts/run_planning.sh`.
- Map building from rosbag or RealSense recording: `bash scripts/run_rosbag_build_map.sh` (combine with `run_realsense_sensor.sh` and `run_realsense_bag_record.sh` to capture data).
- Map-based navigation (perception + planning + map + control + RViz): `bash scripts/run_navigation.sh`.
- 3D Gaussian Splatting map generation: `bash scripts/run_3dgs_generation.sh`.

## Testing
- Targeted runs (use `uv run` if inside venv):
  ```bash
  python tests/test_planning_node.py
  python tests/test_map_node.py
  python tests/test_func.py
  python tests/test_disjointset.py  # or test_disjointset_all.py
  ```
- TRT/model loading sanity: `python tests/test_models_trt.py` (needs GPU + model files).
- Some tests rely on fixtures under `tests/data/` and may be slow because of Numba warmup.

## Notes & Tips for Agents
- Heavy binary/data dirs (`datasets*`, `tinynav_db`, `checkpoints`, `models`, `build`) are large; avoid grepping recursively without `rg --files` filters.
- Code expects ROS 2 message types and TensorRT engines present; guard changes that alter topic names, message fields, or model paths.
- Mapping uses shelve DB (`TinyNavDB`) and pose graph solvers; keep serialization keys and file naming stable.
- Performance-critical functions in `planning_node.py` and `map_node.py` are JIT-compiled; keep signatures/NumPy dtypes unchanged.
- Use ASCII and preserve existing coding style; add concise comments only where logic is non-obvious.
- Style: favor compact code; avoid broad `try/except` wrappers (fail fast). Keep comments sparse—use them mainly to explain design intent or non-obvious decisions.

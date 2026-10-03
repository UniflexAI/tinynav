# Local StartLux decision observer

Evaluation deployment on spark (100.75.206.32), 2026-10-03. Open
http://100.75.206.32:8766/decision, or use the Decision Observer link in Planning Lab.
Set a scene and run perception in Planning Lab, then click Evaluate current observation.
Each click submits one immutable snapshot. There is no automatic inference and no
connection from the observer to cmd_vel, target_pose, or planner configuration.

The request adds navigation.running/collision to the observed WorldState and asks
one choice (continue/wait/suggest_left/suggest_right/replan/uncertain) plus one
noul (stuck). Input, questions, source mode, run metrics, output, latency and errors
are recorded in tinynav_temp/decision_observer/<id>.json. UI supports JSON export.
Requests run in a background thread; one request at a time, HTTP timeout 30s.
A result belongs to its captured timestamp, even if the scene has since changed.
Model probabilities are not calibrated safety or navigation guarantees. Directional
rays do not establish footprint clearance. Full-scene inputs should not be mixed
with observed inputs when comparing results.

## Runtime on spark

- Model: startlux-models/StartLux-Decision-4B-Q8_0-GGUF
- HF revision: 3e98f4efc5b9f1c9c9232cc40f862a3b872cc33f
- Model directory: /home/xiaolefang/workspace/startlux-runtime/model-4b-q8
- Inference code: /home/xiaolefang/workspace/StartLux-Decision
- Inference revision: a9101e918e097b26fea5ead9ed893b823e875e46
- llama.cpp revision: 99b95488cac0f00ce3f05af113a8c1e287753f87
- llama.cpp built with CUDA 13, architecture 121 for GB10, all layers on GPU.
- Two slots, total context 8192 (4096 per question), 6 CPU threads.
- llama-server: http://127.0.0.1:8081
- upstream decision adapter: http://127.0.0.1:8090/v1/systemone
- Start: bash /home/xiaolefang/workspace/startlux-runtime/start.sh
- Logs/PID files: startlux-runtime/{llama,adapter}.{log,pid}
- Stop only these processes: kill "$(cat ~/workspace/startlux-runtime/adapter.pid)"
  and kill "$(cat ~/workspace/startlux-runtime/llama.pid)".
- These are detached processes, not boot services. Restart explicitly after reboot.

The dedicated Python 3.12 venv has transformers 5.18.0 and huggingface_hub 1.33.0.
It reads the existing ComfyUI torch 2.14.0+cu130 packages through a .pth entry;
ComfyUI's environment and process were not modified. This torch dependency must
be preserved or replaced if that environment moves. The GGUF engine does GPU
inference through llama.cpp; torch is used by upstream probability readout.

Code license is Apache-2.0. Downloaded weights are CC BY-NC 4.0; commercial use
requires a separate license from StartLux Labs (contact@startlux.com). See the
model's LICENSE and NOTICE. This deployment is for non-commercial evaluation.

## Validation

20 repeated requests on a paused observed L-turn snapshot, with ComfyUI running:
median 373.35 ms, P95 408.0 ms, range 367.6–420.5 ms; first request 452.5 ms.
GPU process allocation about 4892 MiB. These figures are for two questions and
this snapshot, not all states or the native BF16/CUDA graph implementation.
All distributions passed finite/range/sum checks. Robot XY stayed [0,0] across
20 paused requests. Paused-state stuck probability was ~0.08, but the action
still preferred replan (~0.46): prompts need evaluation, no output is executed.
Raw samples: startlux-runtime/benchmark.json and scenario-smoke.json.
Unit tests cover invalid probabilities, single-flight rejection, snapshot isolation
and persistent logging. Existing NavigationLab tests also pass.

The project's published Jev comparisons are author's benchmarks; they do not
establish equivalence in TinyNav or validate control safety. Next work should
label scenario snapshots, compare choices and calibration, then consider bounded
planner hints only after evaluation.

Smoke evaluations after ~9 seconds of ROS simulation (one snapshot each):

- empty: uncertain, top probability 0.338, stuck 0.040, 372.9 ms.
- dead_end: replan, top probability 0.361, stuck 0.067, 455.6 ms.
- l_turn: replan, top probability 0.322, stuck 0.086, 451.8 ms.

These brief runs were cancelled after evaluation and are not arrival-rate benchmarks.

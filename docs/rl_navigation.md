# RL local navigation

A learned local planner that can replace the trajectory library in `planning_node`. A recurrent
policy reads a robot-centric 2.5D map (BEV) built from the stereo depth and outputs `(v, w)`
directly; a deterministic guard checks every command before it reaches the robot.

Branch: `xiaole/rl-nav`. Code: `tinynav/rl/`, plus the `TINYNAV_PLANNER=rl` hooks in
`tinynav/core/planning_node.py` and `tinynav/platforms/cmd_vel_control.py`.

Status (2026-11-07): works end to end on the robot stack and in the corridor simulator. The default
model is **v7** (`tinynav/models/rl_nav_policy_v7.onnx`), the best one validated in the corridor
simulator. **v12** (`rl_nav_policy_v12.onnx`) is kept in the repo: it is better in the training
simulator, especially in narrow passages, but slower and less goal-following in the corridor (see
[Results](#results), [v12](#v12) and [Open issues](#open-issues)).

## How it works

```
stereo depth + VIO pose
  -> rolling 3D occupancy grid (planning_node, 100x100x40 @ 0.1m)
  -> BEV 3x64x64 @ 0.1m (tinynav/rl/bev.py)      goal / velocity / last action (6)
       height above ground, known, own trail      |
  -> CNN -> GRU (memory) -> actor -> action in [-1, 1]^2
  -> scale to (v, w): v in [-0.3, 1.0] m/s, w in [-0.75, 0.75] rad/s
  -> guard (tinynav/rl/rl_planner.py) -> /planning/rl_cmd -> cmd_vel_control -> /cmd_vel
```

### Observation

- **BEV** (`tinynav/rl/bev.py`), robot control center at the image center, heading up:
  - `height`: elevation of the highest occupied voxel above the ground under the robot (ground =
    lower median of hits 0.3-1.5m around it), capped at the robot's z-band; 0 for cells seen with no
    hit. Raw ground hits only cover rings (one per sampled pixel row), so flat ground reads 0 with or
    without a hit; this removed a sim-to-real gap and the camera mounting height.
  - `known`: any voxel in the column observed.
  - `trail`: the robot's own path over the last 400 ticks (~60s), newest brightest.
- **Vector**: goal in the robot frame / 5, measured velocity (EMA of pose differences) normalized,
  previous action.

### Footprint and guard

`robot_specs.RL_FOOTPRINT` is a 1.0 x 0.5m rectangle centered on the control center: a real size
plus margin covering the robots we deploy on (up to ~0.8 x 0.3m). The training sim, the runtime
guard and the corridor simulator all use it.

The guard (`rl_planner.safe_command`) rolls each command out for 1s and requires >= 10cm between the
footprint outline and the obstacle surface, reading the planner's ESDF with bilinear interpolation
(and subtracting half a cell: the ESDF is center to center). It checks the swept area of turning in
place too. If the policy's command fails it tries, in order: turn in place, turn the other way,
creep forward (0.2 m/s), creep forward turning either way, back off, stop. When the robot is already
closer than the margin, a command passes if it never gets closer (escape). A robot asked to move
that moves < 0.2m / 0.3rad in 10s is "stuck": it stops 3s, resets the GRU memory and retries; the
third time it stops until the next goal. The guard does not depend on the obstacle mask being
dilated (the Pilot's planner does not dilate, `planning_node` does by one cell).

### Training simulator (`tinynav/rl/sim.py`)

Batched on the GPU (256 envs). World: flat ground and boxes, plus scenario generators:

- random boxes and walls (density from the curriculum)
- traps: a wall, or a U-cup around the start, between start and goal
- doorways: a wall across the start-goal line whose opening is a door (width from the
  curriculum), half of them extended into a corridor

Perception is a port of the real pipeline: the depth pixels the planner samples (every 10th) are
rendered analytically, then fused with a torch port of `run_raycasting_loopy` +
`update_occupancy_grid` on a toroidal rolling grid, then `make_bev`. Tests check that it matches the
planner code on identical depth (`tests/test_rl_sim.py`). Half the episodes also have a rear camera,
fused like the Pilot's dual-camera planner (front frame rolls/decays, rear rays are added), so one
policy serves 1- and 2-camera robots. The camera forward offset, height and pitch are randomized.

Robot: unicycle with 1-step latency, acceleration limits and 10% actuation noise; `dt` 0.15s (the
depth rate on the robot). With `guard_in_loop` (default) every command goes through a vectorized copy
of the runtime guard on the true geometry, as on the robot.

Reward per step: progress = new closest approach in geodesic distance (shortest path around
obstacles, training only), novelty bonus for new 0.5m cells, time penalty, clearance penalty within
0.25m, turn-rate penalty, reverse penalty (smaller with a rear camera), action smoothness. Arrival:
+10 + 5 x path efficiency; collision: -10.

### Learning

`tinynav/rl/train_ppo.py`: recurrent PPO (each env's 64-step rollout is one GRU sequence), a
critic that additionally sees privileged geodesic features (training only), a 5-level curriculum
(goal distance, density, trap size, U-cup share, doorway width and share), level-up after a
cooldown when success >= 80%, collisions <= 10% and doorway success >= 60% over the last 10 updates,
LR annealing, time-out bootstrapping, and `best.pt` chosen on a fixed evaluation set
(success - 0.5 x near-miss) rather than the last checkpoint.

Optional behaviour cloning from human-driven recordings (`--demos`): `tinynav/rl/demos.py` rebuilds
exactly the policy's observation from a bag (after `perception_node`) and labels the driver's
`(v, w)` from pose differences, with a hindsight goal 2.2m further along the driven path. Frames
where the 1.0 x 0.5m footprint would come closer than the guard margin are left out (recordings are
from go2 and B2). 11 local bags with narrow sections were processed: ~19.7k frames, 726 narrow,
safe and moving (`tinynav_db/rl_demos/`). Bags were found by screening the RealSense depth of all
recordings for walls on both image edges.

## Usage

All commands run inside the `tinynav-dev` container from `/tinynav` (torch is in the optional `rl`
extra).

```bash
# real BEV from a bag (needs /slam/depth + /slam/odometry_visual)
uv run python -m tinynav.rl.export_bev --bag <bag> --out bev.npz --png bev.png

# train / fine-tune
uv run python -m tinynav.rl.train_ppo --run tinynav_db/rl_runs/<name> --steps 20000000 \
  --init tinynav_db/rl_runs/ppo_v7_bigbody/best.pt --level 3 --lr 1e-4 --lr-end 1e-5 --eval-every 50 \
  [--demos "tinynav_db/rl_demos/*.npz" --bc-coef 1.0 --bc-coef-end 0.1]
tensorboard --logdir tinynav_db/rl_runs --port 6006

# evaluate (hardest level; --trap-prob / --narrow-prob for focused sets) and render GIFs
uv run python -m tinynav.rl.eval_policy --ckpt <best.pt> --envs 512 --seed 2024 --gif eval.gif
uv run python -m tinynav.rl.showcase --ckpt <best.pt> --out tinynav_db/rl_runs/showcase

# demonstrations from a raw stereo bag
bash tinynav/rl/process_demo_bag.sh tinynav_db/rosbags/<bag> <name>

# export for the robot
uv run python -m tinynav.rl.export_onnx --ckpt <best.pt> --out tinynav/models/rl_nav_policy_vN.onnx
cd tinynav/models && make rl_nav [RL_NAV_ONNX=rl_nav_policy_v7.onnx]   # per-arch TensorRT plan
```

### Deployment

Set `TINYNAV_PLANNER=rl` for **both** `planning_node` and `cmd_vel_control` (the app passes its
environment to both). `planning_node` keeps the occupancy grid and ESDF, runs the policy
(`RLNavPolicyTRT`, ~0.07ms on an RTX 3060) and publishes `/planning/rl_cmd`, the rollout on
`/planning/trajectory_path` and the policy input on `/planning/rl_bev`; `cmd_vel_control` executes
`/planning/rl_cmd` with its pause, stale-command, acceleration and minimum-speed guards. The goal is
`/control/target_pose` from `map_node`. `TINYNAV_RL_ENGINE` overrides the engine path. Unset
`TINYNAV_PLANNER` to go back to the trajectory library.

## Results

Training simulator, hardest level, 512 episodes (seed 2024). Columns differ per row as the
evaluation mix grew; "guard" means the guard was applied during evaluation.

| Model | Body | Notes | Success | Collision | Traps | Narrow |
|---|---|---|---|---|---|---|
| v1 | 0.4 x 0.3 | memoryless | 89.6% | 2.9% | 25% (trap mix) | - |
| v4 | 0.4 x 0.3 | GRU, privileged critic, trap curriculum, LR anneal | 92.6% | 2.3% | 86.3% | - |
| v5 | 0.4 x 0.3 | + clearance / turn / efficiency rewards | 92.6% | 2.9% | 83.5% | 68.4% |
| v7 | 1.0 x 0.5 | retrained on the shared RL footprint | 86.3% | 5.3% | 78.5% | 61.1% |
| v7 (guard) | 1.0 x 0.5 | same model, guard on, 35% doorways | 83.8% | 0% | 79.4% | 56.0% |
| v12 (guard) | 1.0 x 0.5 | guard in the loop, doorway curriculum, demos, rear camera | 89.1% | 0% | 83.2% | 71.0% |

Doorway-only scenes (0.8-1.2m doors, guard on): v7 reached 59%, v12 72%.

Corridor simulator (Gazebo reconstruction of a real site, Pilot stack, sim collision = RL footprint,
1 m/s cap; fixed and random spawns, routes over three POIs):

| Model | Runs that localized | Completed | Typical 1->2->3 time |
|---|---|---|---|
| v7, 0.5 m/s cap (before the speed fix) | 6 | 6 | ~220s |
| v7, 1 m/s | 5 | 5 | ~102s, guard intervention ~0% |
| v12, 1 m/s | 3 | 2 | 175s; ~25% slower, detours, one run wandered off |

Random spawns sometimes fail to relocalize within 6 minutes; that is `map_node` relocalization, not
the planner.

## v12

Kept as `tinynav/models/rl_nav_policy_v12.onnx` (from `tinynav_db/rl_runs/ppo_v12_guarded/best.pt`,
18.0M steps) for comparison and as the starting point for the next round. Build it with
`make rl_nav RL_NAV_ONNX=rl_nav_policy_v12.onnx` and set
`TINYNAV_RL_ENGINE=/tinynav/tinynav/models/rl_nav_policy_v12_<arch>.plan`.

How it was trained (on top of v7, which was v5 retrained on the 1.0 x 0.5m footprint):

| Step | Run | Change | Outcome |
|---|---|---|---|
| v8 | `ppo_v8_rear_froze` | rear camera in half the episodes, reverse penalty, 35% doorways, penalty on any command whose 1s rollout came within the guard margin | froze near walls: time-outs 10% -> 20%, stopped at 5M steps |
| v8 | `ppo_v8_rear` | penalty only for commands the guard would refuse, clearance penalty halved, time penalty doubled | still below v7 at 8M (75.8% / 11.3% collisions), stopped; guard penalty turned off |
| v9 | `ppo_v9_demos` | behaviour cloning on ~19.7k frames of human driving (narrow frames x3) | narrow 53% -> ~57%, collisions doubled, stopped |
| v10 | `ppo_v10_geoprog` | progress on geodesic instead of straight-line distance | no change on doorways (56% / 33% time-out), stopped |
| v11 | `ppo_v11_doors` | doorway-width curriculum (1.4-2.0m down to 0.8-1.2m), level-up needs 60% doorway success | used wide doors, still walked round 0.8-1.2m doors |
| v12 | `ppo_v12_guarded` | runtime guard applied in the training sim (from v11 at 5.7M) | doorways 56% -> 71%, success 89.1%, 0 collisions |

Diagnosis that led to v11 / v12: on doorway-only scenes failures were mostly time-outs that never
came within 1.5m of the door (even successes walked round the wall end at 0.8-1.2m). The geodesic
went through the door in 86% of scenes, driving straight through the door center succeeded in 69/80,
and the 0.8m gap was clearly visible in the BEV, so the door was avoided rather than missed: in
training a slightly misaligned approach clipped the frame and ended the episode at -10, while on the
robot the guard would have corrected it.

In the corridor simulator (1 m/s, guard on, same trials as v7): fixed spawn 1->2->3 in 175s (v7
102s) and 3->1->2 in 128s (v7 104s); one random spawn timed out after wandering 74m on its last leg;
median speed 0.36-0.39 m/s vs 0.50 for v7, and legs up to 60% longer. The likely causes are open
issues 1 and 2 below.

## Lessons learned

- **One footprint everywhere.** Training on 0.4 x 0.3 while the guard used 0.6 x 0.3 and the
  simulator a 0.4m disc produced "corner scraping" that no reward could fix.
- **Measure guard distances to the obstacle surface** and do not assume upstream dilation.
- **The guard needs an exit for every geometry**: creeping forward when backed against a wall,
  moving along a wall it is hugging, and stuck retries instead of latching.
- **Match the robot's speed.** With the simulator capped at 0.5 m/s the policy (trained for 1 m/s)
  over-corrected and hugged walls; at 1 m/s guard interventions dropped to ~0%.
- **Penalizing close rollouts teaches freezing.** A penalty on every command whose 1s rollout came
  near a wall made standing still the cheapest action (v8); dropped.
- **Narrow doors were avoided, not missed.** The opening is plainly visible in the BEV and the
  geodesic ran through it, but in training a slightly misaligned approach ended at -10 while a
  detour only cost time. Training with the guard in the loop and a doorway-width curriculum lifted
  doorway success from ~56% to 71%.
- **Pick checkpoints by evaluation**, not the last one; training often degrades late.
- **Simulator gains must be checked in the target setup.** v12 beat v7 in the training simulator
  but not in the corridor.

## Open issues

1. **Goal mismatch.** Training goals are fixed points 2-8m away; on the robot `map_node` gives a
   carrot ~2.2m ahead on the global path that keeps moving. v12 (trained to explore and go around
   obstacles more) does not follow the carrot as well as v7. Next: carrot-following episodes in the
   training simulator (geodesic path + moving 2.2m target).
2. **Speed.** v12 drives ~25% slower than v7, probably from behaviour cloning on cautious human
   driving and training with the guard. Next: lower or drop the BC weight.
3. **Narrow doors (0.8-1.2m)** still time out ~28% of the time; most failures give up on the door
   from a distance.
4. Only one corridor scene was tested; no real-robot runs with v7/v12 yet.

## Files

| Path | Purpose |
|---|---|
| `tinynav/rl/bev.py` | BEV construction shared by sim and robot |
| `tinynav/rl/sim.py` | batched GPU training simulator |
| `tinynav/rl/policy.py` | recurrent actor-critic |
| `tinynav/rl/train_ppo.py` | recurrent PPO, curriculum, BC loss |
| `tinynav/rl/eval_policy.py`, `showcase.py`, `sim_preview.py` | evaluation, GIFs, sim vs real BEV |
| `tinynav/rl/rl_planner.py` | runtime observation, policy call, guard, stuck handling |
| `tinynav/rl/export_bev.py`, `demos.py`, `process_demo_bag.sh` | data from bags |
| `tinynav/rl/export_onnx.py`, `tinynav/models/rl_nav_policy_v*.onnx`, `Makefile` | deployment |
| `tests/test_rl_sim.py`, `tests/test_rl_planner.py` | sim-vs-planner equivalence, guard, rewards |
| `tinynav_db/rl_runs/` (not in git) | training runs; `ppo_v7_bigbody`, `ppo_v12_guarded` are the current candidates |

## v13 experiment and detailed report (2026-11-07)

v13 fine-tunes v12 with a 2.2m moving route target and no BC loss. Training stopped early
after regression; the exported candidate uses best.pt at 2,056,192 steps, not the
9,338,880-step last checkpoint. Matched tests (two seeds, 512 episodes per model/mode)
found moving-target success v7 86.91%, v12 89.45%, v13 92.19%; fixed-goal success
v7 86.13%, v12 89.65%, v13 87.30%. These are NavSim results, not Gazebo or real-robot validation.
The default remains v7.

See [the detailed training and comparison report](rl_navigation_v13_report.md) for
CNN/MLP/GRU inputs and outputs, PPO, rewards, training provenance, paired example
videos, raw results, ONNX validation and target-machine deployment instructions.
The candidate is `tinynav/models/rl_nav_policy_v13.onnx`; build with
`make rl_nav RL_NAV_ONNX=rl_nav_policy_v13.onnx` and select its plan with
`TINYNAV_RL_ENGINE`.

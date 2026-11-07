# RL local navigation

A learned local planner that can replace the trajectory library in `planning_node`. A recurrent
policy reads a robot-centric 2.5D map (BEV) built from the stereo depth and outputs `(v, w)`
directly; a deterministic guard checks every command before it reaches the robot.

Branch: `xiaole/rl-nav`. Code: `tinynav/rl/`, plus the `TINYNAV_PLANNER=rl` hooks in
`tinynav/core/planning_node.py` and `tinynav/platforms/cmd_vel_control.py`.

Status (2026-11-07): works end to end on the robot stack and in the corridor simulator; v7 is the
best model validated in the corridor simulator, v12 is better in the training simulator but not in
the corridor (see [Results](#results) and [Open issues](#open-issues)).

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
4. **Default model.** `make rl_nav` / `RLNavPolicyTRT` currently default to v12
   (`rl_nav_policy_v12.onnx`); v7 (`rl_nav_policy_v7.onnx`) is the better-validated choice for the
   corridor: `make rl_nav RL_NAV_ONNX=rl_nav_policy_v7.onnx` and point `TINYNAV_RL_ENGINE` at its plan.
5. Only one corridor scene was tested; no real-robot runs with v7/v12 yet.

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

# Odometry perturbation experiment

Updated: 2026-11-21. Source base: `efc25a2`; simulation-only changes.

## Model and injection boundary

`tool/stair_odom_noise.py` defines timestamp-indexed position/yaw/pitch errors.
Position is additive in world XYZ. Yaw rotates about world Z; pitch rotates about
the camera optical X axis, preserving the camera optical convention. Correlated
Gaussian errors use an OU sequence sampled every 0.02 s, interpolated by timestamp.
Pitch can additionally include a sinusoid. Errors ramp in over the first second.
A fixed seed and timestamp produce the same error regardless of lookup order or
how frequently the raw/synchronized streams query it.

The headless harness applies the same estimated pose to StairNode's raw-pose
callback, synchronized depth/pose callback, planner and tracking callback. Its
existing callback cadence remains 10 Hz (target timer 2 Hz), so this experiment
does not validate an actual 100 Hz VIO/DDS pipeline. Ground-truth camera pose is
still used to render depth. Physics, gait, world collision geometry and count/
height/shaft/final-stop assertions remain unchanged. This models pose estimation
error; it does not physically shake the camera or add depth noise.

Default is `--odom-noise none`: pose is copied bit-for-bit without perturbation.
The production navigation algorithms, parameters and safety thresholds are not
changed by the noise experiment. The interactive viewer also supports this noise model; see the controls below.

## Parameters and short sweep

The experimental reference (`--noise-scale 1`) is:

| Component | Reference magnitude |
| --- | --- |
| Position correlated noise | 0.01 m standard deviation per world axis |
| Yaw correlated noise | 0.5 degree standard deviation |
| Pitch correlated noise | 0.5 degree standard deviation |
| Periodic pitch | 1 degree peak, 1.7 Hz |
| Correlation time | 0.2 s |

`position`, `yaw`, `pitch` modes enable only their corresponding components;
`combined` enables all. Scale multiplies every error amplitude, including the
pitch sinusoid, but does not change frequency/correlation. These are synthetic
experimental settings, not measured real-world VIO specifications or a certified
tolerance. Initial seed is 7. One run per setting is not a probability estimate.

The first short sweep uses one relative floor (two flights) in the same ten-level
building, flight width 1.2 m and platform depth 1.2 m. Both directions are tested
with disabled noise, individual components at scale 1, and combined at scales
0.5, 1 and 2. Representative full nine-floor follow-ups are selected from those
outcomes. Failure is retained as failure; assertions are not bypassed for a pass.

## Reproduction

Use the prepared isolated container from
[the development guide](stair_floors_development_simulation.md), with ROS sourced,
`ROBOT_TYPE=go2`, `ROS_DOMAIN_ID=79`, `ROS_LOCALHOST_ONLY=1` and Unitree disabled:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_stair_mode.py tests/test_stair_sim_course.py tests/test_stair_odom_noise.py
python -u tool/stair_mode_sim.py --sim-root /simrepo --direction down \
  --levels 10 --floors 1 --flight-width 1.2 --landing-depth 1.2 \
  --odom-noise combined --noise-seed 7 --noise-scale 1 \
  --result-json /simwork/output/noise-down-combined-s1.json
```

Change direction to `up`, mode or scale, and use unique result filenames.
Full tasks use `--floors 9`; `--capture FILE.npz` optionally adds synchronized
video/BEV data, at significantly greater size. All noise parameters have explicit
CLI overrides; use `--help` for their units.

`--result-json` writes success/failure, exception, physical count events,
minimum shaft clearance and stop/hold coordinates. A companion `.trace.npz` records
simulation time, ground-truth and estimated camera transforms, applied errors,
targets, commands, status/count and the noise configuration. Failure snapshots
can occur before a new command is computed, so the last trace row can retain the
previous command/status. The JSON and process exit determine pass/fail.
Arrival hold follows the existing stationary physical hold test; the trace covers
the navigation loop, not extra incoming pose updates during that hold.

The initial module plus existing focused suites pass 49 tests on the 3090.
Tests include exact disabled behavior, shared timestamp lookup, deterministic
seeds, optical/world rotation axes and long-run noise scale/correlation.

## Evidence

3090 experiment directory:
`/home/dm/tinynav-stair-dev/experiments/odom-noise/`.
Each experiment has a log, JSON outcome and trace NPZ. The matrix runner records
actual process exit codes; `short-summary.json` contains the fourteen short cases.
Results and interpretation are appended after the sweep finishes.

## Initial findings (before final-support repair)

The fourteen one-floor runs completed: thirteen passed, one did not arrive within
180 simulated seconds. The failed case was DOWN with position-only noise at
reference scale, seed 7. It counted both flights, stopped with `no_seed` and zero
command on the final platform, and timed out. No fall/contact/shaft assertion was
triggered. This establishes a terminal-support/arrival symptom, not its exact
geometric root cause; the return reason must be inspected before changing support
or arrival constraints. Other noise settings followed different trajectories and
passed, so the strongest passing scale is not a measured robustness limit.

| Short mode, scale | DOWN 1 | UP 1 |
| --- | --- | --- |
| none | PASS | PASS |
| position, 1 | arrival timeout | PASS |
| yaw, 1 | PASS | PASS |
| pitch, 1 | PASS | PASS |
| combined, 0.5 | PASS | PASS |
| combined, 1 | PASS | PASS |
| combined, 2 | PASS | PASS |

With combined scale 0.5 (position std 5 mm/axis, yaw std 0.25 deg, pitch std
0.25 deg plus a 0.5 deg peak sinusoid), DOWN 9 passed all eighteen flights and
stop/hold checks in 379.6 simulated seconds including hold. Minimum independent
shaft clearance was 0.057171 m, compared with the unperturbed regression's
0.129389 m. A passing run can therefore still have materially less clearance.

The initial UP 9 pilot counted all eighteen flights but stayed in `no_seed` with
zero velocity on the final platform. It was manually interrupted at 720.5
simulated seconds (trace/log preserved under `interrupted-pilot/`), rather than
waiting the full default 1620 s budget. That interrupt is not reported as an
automatic safety assertion or timeout. A deterministic rerun uses the explicit
stricter `--max-sim-seconds 450` budget to make this arrival failure reproducible.
The option can only shorten the original task budget; it does not relax any
physical assertion. Disabled-noise behavior and navigation algorithms are unchanged.

```bash
python -u tool/stair_mode_sim.py --sim-root /simrepo --direction up \
  --levels 10 --floors 9 --flight-width 1.2 --landing-depth 1.2 \
  --odom-noise combined --noise-scale .5 --noise-seed 7 \
  --max-sim-seconds 450 --result-json /simwork/output/noise-up9-budget450.json
```

These are single-seed, fixed-scene perturbation experiments. They do not estimate
real VIO noise or field success probability. Next work is to diagnose the final
`no_seed` condition, then test more seeds and calibrate the model using real bags;
no production thresholds are changed as part of this characterization.

The bounded UP 9 rerun reproduced the same final-platform stop and exited 1 at
450.0 simulated seconds: all eighteen flights counted, `arrived=False`, persistent
`no_seed`/zero command. Minimum shaft clearance was 0.089026 m. The previous
720.5 s pilot had the same final pose to the displayed precision. Thus the result
is **not an UP 9 pass**. The 450 s cap is recorded in its JSON; the other cases
used their original budgets. Final outcome files: `short-summary.json`,
`full-summary.json`, `analysis.json` and `down-position-stop.png`. There are 16
completed characterization cases (14 short, 2 full), plus the separate interrupted
pilot. All logs and traces are preserved.

## Interactive viewer

Open the live simulation page. Choose `Odom noise` (`none`, `position`, `yaw`,
`pitch`, `combined`), `Noise scale` (0.5/1/2) and `Noise seed` (7/17/27), then
press `Start / Reset`. Changes apply only to a new task, preventing mixing settings
inside one run. The default remains `none`. `Pause physics` pauses and resumes.
The applied configuration, current XYZ errors in mm, yaw/pitch errors in degrees
and a ten-second error-history plot are shown beside the BEV panels.
The plot is sampled from the same injected pose errors used by navigation.
The 3D robot follows physical ground truth; pose estimation errors distort sensed
BEV and navigation behavior rather than shaking the physical model directly.
Safety assertions remain tied to truth and latch a pause on violation.
Noisy nine-floor UP is not qualified by the existing noiseless scene description.
The viewer uses the same 0.02 s noise grid and 10 Hz pose callbacks, but its own
physics/control loop; headless and interactive trajectories need not be identical.

## Final-support repair and full regressions (2026-11-21)

The final-platform `no_seed` was reproduced in both directions and real generator
snapshots were saved before changing navigation. The DOWN snapshot still had 57
observed reachable level cells, 35 with the existing 0.3 m obstacle clearance;
the UP snapshot had 10 and 3 respectively. The camera-centred lateral restriction
`abs(lateral) < search_clearance` rejected every one of those candidates. This
was a target-strip constraint, not the obstacle clearance measurement. The vehicle
could stop with valid measured platform support to its side and never advance the
remaining 0.4 m step-in distance. It was not simply a missing depth stream.

The finishing target now permits lateral correction onto those observed,
reachable, same-level, forward-facing candidates while retaining the 0.3 m
obstacle clearance, forward progress and bearing checks. The target still passes
through the existing local planner/controller. It does not mark unknown ground
free, accept a lower landing as level, bypass stale inputs, lower step-in progress
or declare arrival based only on counts. Final-count/no-seed handling also cannot
fall through into the ordinary next-flight blind U-turn fallback.

Replaying the saved clouds turns both stable `no_seed` snapshots into `finishing`
with a target on measured level ground. Three focused cases cover offset support,
rejecting a drop and prohibiting a new turn with no seeds. Against the old compute
method, the offset-ground and no-turn tests fail, and drop rejection still passes.
The final focused suite passes **52 tests**. The eleven integration `assert`
expressions are AST-identical to `efc25a2`.

All results below use flight width/platform depth 1.2 m in a ten-level building;
seed 7. Long runs have an explicit 600 s cap that only shortens the default budget;
all complete well before it. Times include the 1.5 s zero-command hold.

| Case | Result | Flights | Simulated seconds | Minimum shaft clearance | Hold drift |
| --- | --- | --- | --- | --- | --- |
| Position-only, scale 1, DOWN 1 | PASS | 2 | 53.6 | 0.145849 m | 13.36 mm |
| Combined, scale 0.5, UP 9 | PASS | 18 | 370.1 | 0.089026 m | 18.08 mm |
| Combined, scale 0.5, DOWN 9 | PASS | 18 | 379.6 | 0.057171 m | 19.63 mm |
| No noise, UP 9 | PASS | 18 | 367.6 | 0.215729 m | 14.54 mm |
| No noise, DOWN 9 | PASS | 18 | 384.1 | 0.129389 m | 6.08 mm |

All five processes exited 0, with ordered count events, final position/height,
zero-command hold, no fall/contact and positive independent shaft clearance.
Combined scale 0.5 remains position std 5 mm per axis, yaw std 0.25 deg, pitch
std 0.25 deg plus a 0.5 deg peak/1.7 Hz sinusoid. This repairs the demonstrated
single-seed terminal issue, not all possible odometry disturbances or real gait.
The noisy DOWN minimum margin is still smaller than the noiseless margin.

Reproduce with the earlier commands, `--floors 9 --odom-noise combined
--noise-scale .5 --noise-seed 7 --max-sim-seconds 600`, once per direction.
The live viewer service has been restarted on this source; choose settings and
Start / Reset for a new task. It is initially paused with noise disabled.

Evidence: `/home/dm/tinynav-stair-dev/experiments/odom-noise-fix/` contains
`fixed-summary.json`, `baseline-summary.json`, `snapshot-analysis.json`,
`final-support-diagnosis.png`, raw logs, trace NPZ and pre-fix state snapshots.
An initial runner collected JSON from the wrong directory although its processes
exited 0; summaries were reconstructed from the actual JSON, PASS logs and recorded
exit codes. `runner-path-diagnostics.json` retains that collection error. Original
pre-fix evidence is preserved in `odom-noise-original-archived/odom-noise/` and
`/tmp/tinynav-odom-noise-evidence.tar.gz`; no evidence was deleted.

## Web recording follow-up

A later recording exposed a pause/resume command gap in the live harness.
It is repaired without lowering noise or navigation thresholds; actual web
UP 9 now passes and matches the offline navigation trace sample for sample.
See [live/offline parity](stair_live_offline_parity.md) for the cause, 57-test
verification and the new video evidence.

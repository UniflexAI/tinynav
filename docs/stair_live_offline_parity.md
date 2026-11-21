# Live / offline stair simulation parity

Updated 2026-11-21. Both the failed recording and the repaired recording use
`combined`, scale `0.5`, seed `7`: XYZ std 5 mm per axis, yaw std 0.25 degrees,
pitch std 0.25 degrees plus a 0.5 degree peak, 1.7 Hz sinusoid. Noise is unchanged.

## Cause

The old live loop cleared `drive` while physics was paused, then reassigned the
published command only in its 10 Hz navigation callback. Resuming between those
callbacks left the gait with zero input for 1-4 ticks (20-80 ms). Offline held
the current published velocity across those 50 Hz gait ticks. Repeated recording
pauses therefore changed physics and the observed final-platform geometry even
with identical noise parameters. The new final `no_seed` was a different trajectory,
not evidence that the offline pass had the same inputs and failed.

Repeated initialization of the same ascending scene is consistent; earlier mount
compensation variations were primarily between different scene directions. The
full paired traces below also rule out initialization/noise differences here.

## Repair

`tool/stair_sim_control.py` applies the newest published command before every gait
tick in both harnesses. Live pause still freezes physics and zeroes drive; the
first resumed tick now uses the current command, including a newly published stop.
Manual drive remains separate. Production robot navigation/control is unchanged.

The live loop now also asserts sequential count heights, final position/height,
zero commands for a 1.5 s physical hold and drift under 0.12 m. Existing shaft,
wall/rail contact and fall checks stay enabled throughout. `ARRIVED · HOLD PASSED`
is displayed only after independent validation, rather than immediately at the
node's arrival flag. The offline harness's eleven independent assertion expressions
are AST-identical to `efc25a2`.

Live runs save traces and outcome JSON under `/simwork/output/live/runs/`, including
resume tick/command events, ground truth and estimated poses, applied errors,
targets, commands, status, counts and the physical outcome. Display-only wall
transparency still uses a copied model and checks that the physics RGBA is unchanged.

## Evidence

A controlled 25 s paired experiment injected pauses at gait ticks
151, 402, 653, 904 and 1151. Old behavior first diverged at the next 10 Hz sample
(3.1 s); maximum position/orientation differences were 0.304885 m / 21.9175 degrees.
The repaired behavior matched uninterrupted baseline exactly in pose and commands.
These capped 25 s runs intentionally do not finish nine floors; their task timeout
is not treated as a successful navigation regression.

The four focused suites pass **57 tests**:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_stair_mode.py tests/test_stair_sim_course.py \
  tests/test_stair_odom_noise.py tests/test_stair_sim_control.py
```

Full offline UP 9 and DOWN 9 pass with 1.2 m flight width/platform depth, ten
levels and the unchanged noise configuration. The actual web UP 9 recording,
including four intermediate pause/resume operations, also passes all eighteen
count transitions, final pose and zero-command hold. Minimum shaft clearance is
0.089026 m. The first 3686 live/offline samples match exactly in timestamps, true
and estimated camera transforms, noise, target (including NaN), velocity and count.
That comparison covers navigation through arrival. The two hold reference poses
are sampled at slightly different phases (live checks each 50 Hz tick; offline
checks after a five-tick group); both independently meet the same hold tolerances.

See [machine-readable evidence](validation/stair_live_offline_parity_2026_11_21.json).
Server logs, traces and comparison scripts are in
`/home/dm/tinynav-stair-dev/experiments/live-offline-fix/`.
New media replaces the upper half of the previous 52 s web recording with a new
26 s ascending run, retaining the original arrived descending recording.

This establishes fixed-scene/single-seed simulation-loop parity. It does not
qualify physical stair gait or establish a general odometry-noise tolerance.

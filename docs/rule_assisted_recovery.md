# Rule-assisted recovery

Branch: `xiaole/rule-assisted-recovery`.

This branch integrates observed rule recovery into `CmdVelControlNode`, the
controller launched by the navigation application. Recovery is enabled by
default on this branch. No model service, key, prompt or model worker is used.
The planner algorithm is unchanged from the original baseline.

## Use on a robot

Use the existing application/navigation startup and select a target as usual.
Do not launch a second controller or a separate recovery velocity publisher.
Recovery runs inside the existing controller and shares its `/nav/active` and
`/nav/paused` gates. Existing teleoperation ownership is retained while
navigation is inactive. Manual emergency stop and pause take priority.

To disable recovery before starting the application/controller:

```bash
export TINYNAV_RULE_RECOVERY=0
```

Or start the controller with `--ros-args -p rule_recovery_enabled:=false`.
This is a startup setting; restart the controller to apply a change.
Inspect progress with:

```bash
ros2 topic echo /navigation/recovery/status
```

Required inputs: synchronized `/slam/depth` and `/slam/odometry_visual`, camera
intrinsics from `/camera/camera/infra2/camera_info`, and the existing
`/control/target_pose`. Camera-frame conventions match the original planner.
Targets use the same raw world coordinates as that planner; the application's
legacy `odom` label is accepted. No new frame transform is inferred. Controllers
that bypass `cmd_vel_control.py` are not integrated by this change.

## Policy and guards

At least six seconds of less than 0.1 m movement triggers a bounded scan.
The scan completes about 360 degrees, then waits for a new depth observation.
Use measured depth rays only; no-return pixels and missing cells stay unknown.
Intrinsics retain principal-point offsets. Clear evidence expires after 30 s;
blocked evidence is retained conservatively. No traversal history marks cells
free. A conservative circular footprint includes body/control offsets and a
5 cm additional recovery margin.

Every recovery candidate must have zero unknown and blocked footprint cells
along its entire sampled sequence. Prefer a unique progressive pivot. Otherwise
keep the longest observed retreat on each side; prefer fewer prior attempts,
then longer retreat, then fixed ID. If no retreat is eligible, use an observed
progressive pivot. No eligible action means no recovery motion.

Recovery uses at most 0.2 m/s translation and 0.4 rad/s rotation, also bounded by
robot limits. Ordinary planner motion retains its original limits. Short probes
keep a 0.6 m distance at reduced speed. A longer detour candidate retreats 3 m,
turns 90 degrees and probes 1.8 m to avoid returning immediately into a trap.
It is eligible only when the whole swept footprint is already observed clear.

Check the whole remaining stage against updated observations every control
cycle. Abort on unknown/blocked footprint, sensor age over 0.5 s, controller
timer gap over 0.25 s, excessive pose deviation, or execution timeout. Scanning
requires an observed clear rotation footprint, aborts after 25 s, and permits
at most 2 cm translation. Three scans maximum per target, cooldown eight seconds.
Paused/stopped navigation releases recovery immediately. Target changes over
0.25 m reset recovery and observations; small localization corrections preserve
the current session. A new navigation activation resets the recovery
session. No temporary target or mission-goal change is used.

Near the original goal (within 0.65 m), a six-second stall may trigger one
observed, low-speed finishing attempt: align toward the goal, then advance at
the configured minimum executable speed. It requires an observed clear rotation
footprint and forward swept footprint, fresh sensors, and unchanged navigation
activation. Limit this attempt to 15 s; it never invents a temporary goal. This
handles the original controller stopping tiny path-following commands outside
the 0.35 m arrival threshold. It does not change the planner algorithm.

Recovery owns velocity publication only while recovering; otherwise the same
controller resumes its original path following. Arrival within 0.35 m of the
original target holds a stop until a new target/session.

## Websim

Inside the ROS-enabled container:

```bash
bash scripts/run_rule_assisted_web.sh
```

Defaults: localhost ROS domain 216 and loopback HTTP port 8774. Do not start a
second instance on an occupied domain/port. Forward the port and open `/decision`.
Choose one of five scenes and rule on/off. Default run limit is 180 s to cover
the conservative recovery speeds and subsequent path following. The web simulator now launches the
same `cmd_vel_control.py` used by the robot, not the previous simulator-only
controller. Each run restarts planner/control and resets the same scene/pose.
The simulator mirrors status from `/navigation/recovery/status`; it does not
run a second recovery policy or override velocity.

Comparison records are separated by native control version and configuration.
Old simulator-controller results are not presented as this version's results.
Historical model experiments remain in the observer branch and local evidence
archives, not in this branch's active source tree.

## Validation

Targeted tests cover candidate coverage, measured depth behavior, stale sensors,
pause/inactivity, scan budget, target resets, long-detour footprint rejection,
selection by length, ROS message types and single-controller arbitration.
Live simulation results are recorded separately in
`docs/rule_recovery_native_results.md`. This branch has not yet been tested on
physical hardware; deployment requires the chosen robot and matching robot
geometry/calibration. Static-simulation zero collisions is not a physical test.

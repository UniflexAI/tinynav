# Multi-stage recovery sequences

This policy is limited to explicitly enabled, loopback-only synthetic websim model experiments. Normal websim remains off. No robot command is published by the model consumer.

Eight bounded proposals comprise left/right 60-degree pivot + 0.6m probe, or 0.6m/1.2m retreat + 90-degree turn + 0.6m probe, or 1.8m retreat + 135-degree turn + 0.6m probe. Linear speed is capped at 0.3m/s and angular speed at 0.6rad/s, also respecting configured robot limits. Longest nominal sequence is about 12 seconds. The original planner runs throughout and resumes control after completion or rejection.

Proposals use goal coordinates, robot configuration and accumulated measured depth occupancy. They do not use synthetic object geometry. A conservative circular footprint sample rejects measured blocked cells and reports unknown fraction; unknown is not asserted clear. The local model chooses one offered sequence or keep_current/request_new_candidates/uncertain. No deterministic fallback chooses a sequence when the model abstains.

Every stage is checked from the actual current pose with the simulator's box collision function over the remaining segment at 0.05s intervals. Every integration step also checks its swept footprint at 0.05s intervals, catching newly introduced obstacles. A blocked stage or step stops recovery and returns control to the planner. These execution checks use simulator ground truth and are not a deployable perception-based safety layer. Discrete sampling is not a continuous collision proof. Imported map volumes are unsupported and refused.

Selection requires unchanged scene generation, response age <=8s, source pose distance <=0.2m/yaw difference <=15 degrees, and a planning report received within 0.5s. Execution also aborts on stale planning reports, episode deadline, pause/end of run, or reset. Each stage uses accumulated integration time with the final step clipped to remaining duration. An additional five-second wall-time slack guards delayed timers.

The last 24 attempted sequences retain source pose, outcome, duration, end pose and goal progress. After returning to the planner for five seconds, net progress is measured again and included in subsequent model input. The same sequence is suppressed within 0.6m and 35 degrees of its previous source pose. Scene reset clears this memory. A completed sequence is an executed command sequence, not a successful escape. The existing recovery progress metric remains separate from arrival.

The candidate library is expanded relative to the previous single-action experiment; previously attempted local options are removed. `request_new_candidates` remains an abstention: it does not generate additional templates or invoke unconstrained model-generated trajectories. If all eight local templates are exhausted or measured-blocked, the model can abstain and the original planner continues. Persistent failures will need additional templates or a different escape trigger rather than repeating the same sequence.

Reproduce the 20-run comparison on spark:

```sh
bash scripts/start_closed_loop_services.sh
python3 scripts/run_closed_loop_experiment.py --repeats 5 --timeout 120 --output tinynav_temp/closed_loop_strategies_20261003
docker exec tinynav-dev python3 /tinynav/scripts/stop_closed_loop_services.py
```

Do not launch duplicate servers. The four services retain ROS domains 220..223 and ports 8770..8773. The dated folder contains every run's scene, metrics, pose samples and strategy events. The results page `/static/closed-loop.html` allows selecting this experiment or the previous 0.75-second-action experiment. Both are read-only views.

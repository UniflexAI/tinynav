# Isolated model-assisted closed-loop experiment

The default websim remains in `off` mode. Only explicitly configured loopback servers with nonzero ROS domains accept experimental pose overrides. The model never publishes robot commands.

Run on spark from the repository root:

```sh
bash scripts/start_closed_loop_services.sh
python3 scripts/run_closed_loop_experiment.py --repeats 5 --timeout 120 --output tinynav_temp/closed_loop_20261003
docker exec tinynav-dev python3 /tinynav/scripts/stop_closed_loop_services.py
```

Servers 8770..8773 use ROS domains 220..223 and ROS_LOCALHOST_ONLY=1. Do not launch duplicate servers. L-turn and dead-end each have baseline/model groups, with five 120-second repeats. Four groups run concurrently against a shared local model, so latency includes contention. No random seed guarantees identical ROS scheduling.

Model calls begin after a motion-derived stall: at least 3 seconds of recent history, displacement below 0.1m and yaw change below 15 degrees. Calls are single-flight per server with a two-second cooldown. The planner continues while inference is pending. Responses older than eight seconds are rejected. Execution requires a fresh report (received within 0.5 seconds), unchanged scene generation, position within 0.2m and yaw within 15 degrees of the source report, an offered candidate, stable velocity, and no sampled footprint collision.

Accepted commands override only synthetic pose integration for 0.75 seconds, then the planner resumes. Timing is discretized by simulator ticks; this is a simulation experiment, not a real-time controller guarantee. Finite reverse-gate penalties may be tested; collision-rejected candidates may not. Confidence threshold is zero for this experiment. Low confidence is retained in raw events. Observer component records remain marked observer_only because that component does not publish commands; the separate experimental consumer records every actual application.

Recovery is separate from arrival: following the first detected stall, net goal progress must reach 0.25m and remain for five seconds without collision. A partial improvement below that threshold is not recovery. Raw records contain poses, configuration, baseline definitions, requests, applied commands, response confidence, latency, and fallback reasons.

The read-only result page is `/static/closed-loop.html`; `/api/experiment/results` reads the dated output folder. It does not start experiments. A run may end while a request is pending; request counts can therefore exceed completed response counts.

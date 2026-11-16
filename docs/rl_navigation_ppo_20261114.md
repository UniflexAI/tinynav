# Spark PPO candidate for device evaluation

Build the engine inside the target container with its installed TensorRT.
Run these commands from the TinyNav checkout inside the container.

```bash
git lfs pull --include=tinynav/models/rl_nav_policy_ppo_20261114.onnx
trtexec --onnx=tinynav/models/rl_nav_policy_ppo_20261114.onnx \
  --saveEngine=tinynav/models/rl_nav_policy_ppo_20261114.plan --noTF32
export TINYNAV_PLANNER=rl
export TINYNAV_RL_ROUTE_GUIDANCE=1
export TINYNAV_RL_ENGINE="$PWD/tinynav/models/rl_nav_policy_ppo_20261114.plan"
```

Apply the exports to the application startup environment for both the
planner and velocity controller. Set `TINYNAV_PLANNER=library` to switch
back to the trajectory library.

The candidate uses live occupancy, global route guidance, and swept
footprint clearance checks. Spark evaluation used a local four-cell
BEV correction and corrected simulator collision assets. Device behavior
requires separate validation. The ONNX SHA256 is
`84920b3823ec2140b7b04c22363d83c3890ab0a2360e380ee8503ae9d3993235`.

The PPO guard accepts these startup environment variables in metres:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `TINYNAV_PPO_FRONT_M` | 0.50 | Distance from the configured control center to the front edge |
| `TINYNAV_PPO_REAR_M` | 0.50 | Distance from the configured control center to the rear edge |
| `TINYNAV_PPO_HALF_WIDTH_M` | 0.25 | Distance from the configured control center to either side |
| `TINYNAV_PPO_CONTROL_X_M` | 0.00 | Shift from the robot profile's control center, positive forward |
| `TINYNAV_PPO_CONTROL_Y_M` | 0.00 | Shift from the robot profile's control center, positive left |

Start with measured extents relative to the chosen control center. For the
Spark Go2 replay geometry, set the front, rear and half width to 0.25,
0.35 and 0.15 m with zero additional shift. The active geometry appears
in the startup log. The 0.10 m guard margin and the existing near-obstacle
escape rule apply to the configured outline.

`PPO clearance blocked` indicates a zero command while the planner
checks subsequent obstacle maps for a safe motion. `PPO stuck` indicates
the planner published nonzero commands without sufficient observed
translation or rotation and could not recover. Compare `/planning/rl_cmd`
with `/cmd_vel` to see what the controller forwarded.

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

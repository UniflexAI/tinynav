# v16 navigation candidate

Retained experimental baseline for device evaluation, exported from checkpoint
step 1,851,392. This is the archived v16, not the later dynamic/stair experiments.
Default policy selection is unchanged.

## Model and device setup

- Model: `tinynav/models/rl_nav_policy_v16.onnx` (Git LFS).
- SHA256: `095a26ad896a78805f56c142efc03defad6b178af1b50107711d66eea2ef8483`.
- Inputs: float32 `bev[1,3,64,64]`, `vec[1,6]`, recurrent state `h[1,256]`.
- Outputs: `action[1,2]`, `h_out[1,256]`. Initialize recurrent state to zero
  for a new episode/goal; carry it between inference calls.
- Existing BEV/vec/action scaling contract is unchanged. Set the policy through
  the existing RL planner and engine override, not by bypassing the guard.

```bash
git lfs pull --include=tinynav/models/rl_nav_policy_v16.onnx
trtexec --onnx=tinynav/models/rl_nav_policy_v16.onnx \
  --saveEngine=tinynav/models/rl_nav_policy_v16.plan --noTF32
# Set these in the application's startup environment, then restart the app:
export TINYNAV_PLANNER=rl
export TINYNAV_RL_ENGINE=/tinynav/tinynav/models/rl_nav_policy_v16.plan
```

Build TensorRT on the target device with its installed TensorRT version. The
home x86 GPU engine is not published as a Jetson-compatible artifact. FP16/INT8
are not validated here. ONNX checker passed. Home FP32/noTF32 TensorRT parity
was checked over128 steps: maximum action difference6.56e-7, hidden5.22e-6.

## Evidence and limits

Independent NavSim tests of the retained checkpoint:

| Test | Result |
|---|---|
| L corners |16/16 passed |
| Near rear goals |16/16 arrived;15/16 strict direct reverse |
| Far rear goals,1.5/1.8m corridors |8/8 arrived;6/8 strict turn-first |
| Standard scenes |30/32 arrived |

These small independent groups had no collisions; model selection evaluation
also recorded1 collision/256, so zero collision is not a general guarantee.
Far rear goals in1.1/1.3m corridors timed out.

Home Gazebo truth-guided local-goal tests completed a clockwise ring and a
separately initialized counterclockwise ring. Immediate reverse-direction start
stalled at the initial180-degree turn. These are not full localization/POI or
real-device acceptance tests. Dynamic-obstacle avoidance and stairs are NOT
validated capabilities of v16. No real-device v16 acceptance is claimed.

The archived export identity and limitations are also recorded in
`rl_navigation_v16_manifest.json`.

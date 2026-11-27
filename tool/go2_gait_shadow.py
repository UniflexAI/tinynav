"""Run the Go2 v3 gait against live feedback without publishing motor commands."""

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

SDK_JOINTS = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
PERIOD = 0.02
CMD_TIMEOUT = 0.25
STATE_TIMEOUT = 0.10


class CommandGate:
    """Latch a fault until the process is restarted and fresh inputs arrive."""

    def __init__(self):
        self.reason = None
        self.started = False

    def check(self, now, command_at, state_at, estop=False):
        if estop:
            self.reason = self.reason or "estop"
        if self.reason is not None:
            return False
        if command_at is None or state_at is None:
            return False
        if now - state_at >= STATE_TIMEOUT:
            self.reason = "feedback_timeout"
        elif now - command_at >= CMD_TIMEOUT:
            self.reason = "command_timeout"
        self.started = self.reason is None
        return self.started


class ShadowGait:
    def __init__(self, policy):
        self.policy = policy
        self.contract = policy.contract
        names = self.contract.joint_names
        groups = self.contract.action_groups
        if (policy.uses_depth or len(names) != 12 or set(names) != set(SDK_JOINTS)
                or len(groups) != 1 or groups[0].kind != "position"
                or tuple(groups[0].joints) != tuple(names)):
            raise ValueError("Expected a 12-joint Go2 blind position policy")
        self.policy_from_sdk = [SDK_JOINTS.index(name) for name in names]
        self.sdk_from_policy = [names.index(name) for name in SDK_JOINTS]
        self.clock = self.contract.GaitClock()
        self.heading = self.contract.HeadingError()
        self.history = self.contract.ProprioHistory()
        self.last_action = np.zeros(12, dtype=np.float32)
        self.step = 0

    def tick(self, command, q, dq, quaternion, gyro):
        arrays = [np.asarray(values, dtype=np.float32) for values in (command, q, dq, quaternion, gyro)]
        if [a.shape for a in arrays] != [(3,), (12,), (12,), (4,), (3,)]:
            raise ValueError("Invalid Go2 feedback dimensions")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Non-finite Go2 feedback or command")
        command, q, dq, quaternion, gyro = arrays
        if abs(np.linalg.norm(quaternion) - 1.0) > 0.05:
            raise ValueError("Invalid Go2 IMU quaternion")
        command = self.contract.shape_command(command)
        phase = self.clock(command, march=False) if self.clock is not None else None
        heading = self.heading(quaternion, command[2]) if self.heading is not None else None
        order = self.policy_from_sdk
        obs = self.contract.build_proprioception(
            gyro, quaternion, command, q[order], dq[order], self.last_action,
            self.step, heading_error=heading, gait_phase=phase,
        )
        self.history.append(obs)
        action = self.policy(obs, self.history.array, None)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Invalid gait action")
        self.last_action[:] = action
        self.step += 1
        offset = self.contract.default_joint_pos
        scale = self.contract.action_groups[0].scale
        return (offset + np.clip(scale * action, -2.0, 2.0))[self.sdk_from_policy]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iface", required=True)
    parser.add_argument("--sim-dir", type=Path, required=True, help="Checked-out tinynav-sim/mujoco")
    parser.add_argument("--seconds", type=float, default=15.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sim_dir.resolve()))
    from sim.policy import PieOnnxPolicy
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Twist_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    onnx = args.sim_dir.resolve() / "assets/policy/go2_blind_v3.onnx"
    gait = ShadowGait(PieOnnxPolicy(onnx, provider="cpu"))
    ChannelFactoryInitialize(0, args.iface)
    lock = threading.Lock()
    state = command = None
    last_tick = None

    def on_state(message):
        nonlocal state, last_tick
        sample = (
            [float(m.q) for m in message.motor_state[:12]],
            [float(m.dq) for m in message.motor_state[:12]],
            list(message.imu_state.quaternion), list(message.imu_state.gyroscope),
        )
        with lock:
            tick = int(message.tick)
            if tick != last_tick:
                last_tick = tick
                state = (time.monotonic(), sample)

    def on_cmd(message):
        nonlocal command
        with lock:
            command = (time.monotonic(), [message.linear.x, message.linear.y, message.angular.z])

    low = ChannelSubscriber("rt/lowstate", LowState_)
    twist = ChannelSubscriber("rt/cmd_vel", Twist_)
    low.Init(on_state, 1)
    twist.Init(on_cmd, 1)
    print(json.dumps({"mode": "read_only", "sdk_order": SDK_JOINTS,
                      "policy_order": gait.contract.joint_names}), flush=True)
    gate = CommandGate()
    started = None
    next_tick = time.monotonic()
    deadline = next_tick + args.seconds
    while next_tick < deadline:
        now = time.monotonic()
        with lock:
            sample, desired = state, command
        q_target = None
        if sample is not None:
            command_at = desired[0] if desired is not None else None
            if gate.check(now, command_at, sample[0]):
                q_target = gait.tick(desired[1], *sample[1]).tolist()
            if started is None or now - started >= 0.20:
                print(json.dumps({"feedback_age_s": now - sample[0], "q_sdk": sample[1][0],
                                  "dq_sdk": sample[1][1], "q_target_sdk": q_target,
                                  "fault": gate.reason}), flush=True)
                started = now
        next_tick = max(now, next_tick + PERIOD)
        time.sleep(max(0.0, next_tick - time.monotonic()))
    if state is None:
        raise SystemExit("No Go2 low-state feedback received")


if __name__ == "__main__":
    main()

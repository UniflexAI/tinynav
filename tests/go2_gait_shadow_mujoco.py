"""Validate shadow inference and the command gate against the Go2 MuJoCo plant."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from tool.go2_gait_shadow import CMD_TIMEOUT, PERIOD, SDK_JOINTS, CommandGate, ShadowGait


def main(sim, reference_xml):
    sys.path.insert(0, str(sim))
    import mujoco
    from sim.keyboard import ConstantCmd
    from sim import plant
    from sim.plant import build_flat_model, reset_to_spawn
    from sim.policy import PieOnnxPolicy
    from sim.runtime import PieRuntime
    import xml.etree.ElementTree as ET

    def collision_model(_):
        xml = ET.fromstring(plant.GO2_PIE_XML.read_text())
        for node in xml.iter():
            for child in list(node):
                if child.tag == "mesh" or (child.tag == "geom" and child.get("mesh")):
                    if child.tag == "geom":
                        assert child.get("class") == "visual"
                    node.remove(child)
        spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding="unicode"))
        plant._configure_robot_spec(spec)
        return spec

    plant._load_robot_spec = collision_model
    model, b = build_flat_model()
    lookup = [b.joint_names.index(joint) for joint in SDK_JOINTS]
    qids = b.joint_qpos_adr[lookup]
    vids = b.joint_dof_adr[lookup]
    actuators = b.actuator_ids[lookup]
    reference = {j.attrib["name"]: np.fromstring(j.attrib["axis"], sep=" ")
                 for j in ET.parse(reference_xml).iter("joint")
                 if j.attrib.get("name") in SDK_JOINTS and "axis" in j.attrib}
    assert set(reference) == set(SDK_JOINTS)
    for name in SDK_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        np.testing.assert_allclose(model.jnt_axis[joint], reference[name])
    onnx = sim / "assets/policy/go2_blind_v3.onnx"

    for scenario in ("command_timeout", "feedback_timeout", "estop"):
        data = mujoco.MjData(model)
        reset_to_spawn(model, data, b)
        cmd = ConstantCmd(0)
        rt = PieRuntime(model, b, data, PieOnnxPolicy(onnx, provider="cpu"), None, cmd)
        shadow = ShadowGait(PieOnnxPolicy(onnx, provider="cpu"))
        gate = CommandGate()
        root = b.root_qpos_adr
        fault = None
        command_at = state_at = 0.
        for i in range(200):
            now = i * PERIOD
            if i < 100 or scenario != "command_timeout":
                command_at = now
            if i < 100 or scenario != "feedback_timeout":
                state_at = now
            permitted = gate.check(now, command_at, state_at, estop=scenario == "estop" and i == 100)
            if not permitted and fault is None:
                fault = i
                assert gate.reason == scenario
            if fault is not None:
                assert not permitted
            cmd.vx = 0.0 if not permitted or i < 50 else .25
            observation = (data.qpos[qids], data.qvel[vids],
                           data.qpos[root+3:root+7], data.sensordata[b.gyro_adr:b.gyro_adr+3])
            target = shadow.tick([cmd.vx, 0., 0.], *observation)
            rt.tick()
            np.testing.assert_allclose(target, data.ctrl[actuators], atol=1e-6)
            if i == 190:
                tail = data.qpos[root:root+2].copy()
        assert fault is not None
        assert not gate.check(4.0, 4.0, 4.0)
        tail_motion = float(np.linalg.norm(data.qpos[root:root+2] - tail))
        assert tail_motion < .03, (scenario, tail_motion)
        assert not rt.is_fallen()
        print(json.dumps({"fault": scenario, "fault_at_s": fault * PERIOD,
                          "last_9_ticks_displacement_m": tail_motion, "joint_targets_match": 12}))
    print("All 12 MuJoCo axes match the Unitree Go2 model")
    print("Model gate drill simulates a zero velocity request after each latched fault")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim", type=Path)
    parser.add_argument("reference_xml", type=Path)
    args = parser.parse_args()
    main(args.sim.resolve(), args.reference_xml)

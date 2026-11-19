"""Exercise DOWN 1 floor through stair_node, planning_node, and cmd_vel_control.

Requires the Go2 MuJoCo runtime and the stair simulation course builder.
All ROS publishers stay on localhost in the test container.
"""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from std_msgs.msg import Bool

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sim-root", type=Path, required=True)
parser.add_argument("--course-script", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.sim_root / "mujoco"))
from sim import plant
from sim.keyboard import ConstantCmd
from sim.policy import PieOnnxPolicy
from sim.robots import get_robot
from sim.runtime import PieRuntime, DepthCamera

from tinynav.core import stair_node, planning_node
from tinynav.core.math_utils import np2msg
from tinynav.platforms import cmd_vel_control

spec = importlib.util.spec_from_file_location("course", str(args.course_script))
course = importlib.util.module_from_spec(spec)
spec.loader.exec_module(course)
robot = get_robot("go2")
model, binding, surfaces, walls = course.build_world(plant, robot, 2, mapping_rig=False)
data = mujoco.MjData(model)
plant.reset_to_spawn(model, data, binding, pos=course.SPAWN, yaw_deg=7.)
drive = ConstantCmd(0., 0.)
runtime = PieRuntime(model, binding, data, PieOnnxPolicy(Path(robot.policy_path("perception")), provider="cpu"), None, drive)
for _ in range(50):
    runtime.tick()
calibration = course.FrontCameraBEV(runtime, profile="ros-left")
nav_camera = DepthCamera(model, runtime.b.depth_camera_id, 640, 544)
K = np.array([[303.10,0.,269.37],[0.,303.10,314.67],[0.,0.,1.]])
rclpy.init()
args = SimpleNamespace(camera_height=.4669, direction="down", floors=1,
                       landings_per_floor=2, turn="auto", memory=None, rate=2.)
stair = stair_node.StairNode(args)
planner = planning_node.PlanningNode()
control = cmd_vel_control.CmdVelControlNode()
clock = SimpleNamespace(monotonic=lambda: float(data.time))
for module in (stair_node, planning_node, cmd_vel_control):
    module.time = clock
stair.K = planner.K = K
control._on_nav_active(Bool(data=True))
def status(msg):
    planner._on_stair_status(msg)
    control._on_stair_status(msg)
    if msg.data != status.last:
        print(round(data.time,2), msg.data, flush=True)
        status.last = msg.data
status.last = None
stair.status_pub = SimpleNamespace(publish=status)
stair.target_pub = SimpleNamespace(publish=planner.target_pose_callback)
stair.poi_change_pub = SimpleNamespace(publish=planner.poi_change_callback)
planner.path_pub = SimpleNamespace(publish=control.path_callback)
outputs = []
control.cmd_pub = SimpleNamespace(publish=outputs.append)
bridge = CvBridge()
min_clearance = np.inf
wall_ids = {i for i in range(model.ngeom)
            if model.geom(i).name.endswith("_wall") or "rail" in model.geom(i).name}
try:
    for tick in range(1800):
        mujoco.mj_camlight(model,data)
        camid = runtime.b.depth_camera_id
        T = np.eye(4)
        T[:3,:3] = data.cam_xmat[camid].reshape(3,3) @ np.diag((1.,-1.,-1.))
        T[:3,3] = data.cam_xpos[camid]
        pose = np2msg(T, stair.get_clock().now().to_msg(), "world", "camera")
        pose.header.stamp.sec = int(data.time)
        pose.header.stamp.nanosec = int((data.time % 1)*1e9)
        depth = bridge.cv2_to_imgmsg(nav_camera.capture(data), encoding="32FC1")
        depth.header = pose.header
        stair.odom_callback(pose)
        stair.depth_callback(depth,pose)
        control.pose_callback(pose)
        if tick % 5 == 0:
            stair.timer_callback()
        with contextlib.redirect_stdout(io.StringIO()):
            planner.sync_callback(depth,pose)
        control.cmd_timer_callback()
        command = outputs[-1] if outputs else None
        drive.vx = command.linear.x if command else 0.
        drive.wz = command.angular.z if command else 0.
        if tick%30 == 0:
            print("pose",round(data.time,2),runtime.base_pos().round(3),round(drive.vx,3),round(drive.wz,3),flush=True)
        for _ in range(5):
            runtime.tick()
            assert not runtime.is_fallen(), ("fallen",runtime.base_pos())
            for contact in data.contact[:data.ncon]:
                assert not wall_ids.intersection((contact.geom1, contact.geom2)), ("wall contact",runtime.base_pos())
            q = data.qpos[binding.root_qpos_adr:binding.root_qpos_adr+7]
            clearance = course.shaft_clearance(q[0],q[1],course.yaw_of(q[3:]))
            min_clearance=min(min_clearance,clearance)
            assert clearance>0, ("shaft",q[:3])
        if stair.arrived:
            assert stair.gen.landings==2 and drive.vx==drive.wz==0
            stop = runtime.base_pos().copy()
            assert stop[0] < -.5 and abs(stop[2] + 2.88 - robot.spawn_drop) < .12, stop
            for _ in range(75):
                runtime.tick()
                assert not runtime.is_fallen(), ("fallen at stop",runtime.base_pos())
            hold = runtime.base_pos().copy()
            assert np.linalg.norm(hold[:2] - stop[:2]) < .12
            print("PASS DOWN 1 floor", stop, "hold", hold, "shaft clearance", min_clearance, flush=True)
            break
    else:
        raise RuntimeError(("did not arrive",runtime.base_pos(),stair.gen.landings))
finally:
    for n in (stair, planner,control):
        n.destroy_node()
    rclpy.shutdown()
    nav_camera.close()
    runtime.close()

"""Exercise finite DOWN floors through stair_node, planning_node, and cmd_vel_control.

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
parser.add_argument("--capture", type=Path)
parser.add_argument("--floors", type=int, choices=range(1, 6), default=1)
args = parser.parse_args()
capture_path = args.capture
floors = args.floors
flights = floors * 2
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
model, binding, surfaces, walls = course.build_world(plant, robot, flights, mapping_rig=False)
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
args = SimpleNamespace(camera_height=.4669, direction="down", floors=floors,
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
last_landing = 0
landing_events = []
wall_ids = {i for i in range(model.ngeom)
            if model.geom(i).name.endswith("_wall") or "rail" in model.geom(i).name}
frames = []
capture_times, capture_poses, capture_maps = [], [], []
capture_targets, capture_paths, capture_commands, capture_status = [], [], [], []
capture_landings, capture_clearance = [], []
latest_result = {}
original_compute = stair.gen.compute


def record_compute(*values, **options):
    result = original_compute(*values, **options)
    latest_result.clear()
    latest_result.update(result)
    points = np.concatenate([p for _, p in stair.gen.frames])
    _, _, observed, _ = build_height_map(
        points, values[0][:2, 3], values[0][2, 3], stair.gen.cfg)
    latest_result["observed"] = observed
    return result


def record_path(msg):
    latest_path.clear()
    latest_path.extend((p.pose.position.x, p.pose.position.y) for p in msg.poses)
    control.path_callback(msg)


def record_frame():
    q = data.qpos[binding.root_qpos_adr:binding.root_qpos_adr + 7].copy()
    yaw = course.yaw_of(q[3:])
    offset = (31.5 - np.arange(64)) * .08
    x, y = np.meshgrid(offset, offset, indexing="ij")
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.stack((q[0] + c * x - s * y, q[1] + s * x + c * y), axis=-1)

    def sample(grid, origin, resolution, fill=0):
        ij = np.floor((world - origin) / resolution).astype(int)
        inside = ((ij >= 0) & (ij < grid.shape)).all(axis=-1)
        ix, iy = np.moveaxis(np.clip(ij, 0, np.array(grid.shape) - 1), -1, 0)
        return np.where(inside, grid[ix, iy], fill)

    res = latest_result
    local = np.zeros((4, 64, 64), dtype=np.float32)
    if res:
        local[0] = sample(res["height"], res["origin"], stair.gen.cfg.resolution, np.nan)
        local[1] = sample(res["observed"], res["origin"], stair.gen.cfg.resolution)
        local[2] = sample(res.get("reachable", res["free"]), res["origin"], stair.gen.cfg.resolution)
    if latest_mask:
        mask = latest_mask[-1]
        grid = np.asarray(mask.data).reshape((mask.info.height, mask.info.width), order="F")
        origin = np.array((mask.info.origin.position.x, mask.info.origin.position.y))
        local[3] = sample(grid > 0, origin, mask.info.resolution)
    camera.lookat[:] = q[:3] + (0., 0., -.16)
    camera.azimuth = 277. if q[1] > .7 else 105.
    camera.elevation = -45.
    renderer.update_scene(data, camera=camera, scene_option=display)
    frames.append(renderer.render().copy())
    capture_times.append(float(data.time) - start_time)
    capture_poses.append(q)
    capture_maps.append(local.astype(np.float16))
    target = planner.target_pose
    capture_targets.append(np.full(2, np.nan) if target is None else target[:2].copy())
    path = np.full((32, 2), np.nan)
    if latest_path:
        path[:min(32, len(latest_path))] = latest_path[:32]
    capture_paths.append(path)
    capture_commands.append((drive.vx, drive.wz))
    capture_status.append(status.last or "")
    capture_landings.append(stair.gen.landings)
    capture_clearance.append(min_clearance)


latest_mask, latest_path = [], []
renderer = None
start_time = float(data.time)
if capture_path:
    from tinynav.core.stair_target import build_height_map
    stair.gen.compute = record_compute
    planner.obstacle_mask_pub = SimpleNamespace(publish=lambda msg: latest_mask.append(msg))
    planner.path_pub = SimpleNamespace(publish=record_path)
    renderer = mujoco.Renderer(model, height=400, width=720)
    camera = mujoco.MjvCamera()
    camera.distance = 2.3
    display = mujoco.MjvOption()
    display.geomgroup[1] = 0
    display.geomgroup[3] = 1
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
        assert stair.gen.landings in (last_landing, last_landing + 1), (last_landing, stair.gen.landings)
        if stair.gen.landings != last_landing:
            last_landing = stair.gen.landings
            base = runtime.base_pos().copy()
            assert abs(base[2] + 1.44 * last_landing - robot.spawn_drop) < .18, ("count height", last_landing, base)
            landing_events.append((float(data.time) - start_time, last_landing, *base))
            print("COUNT", landing_events[-1], flush=True)
        control.pose_callback(pose)
        if tick % 5 == 0:
            stair.timer_callback()
        latest_path.clear()
        latest_mask.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            planner.sync_callback(depth,pose)
        control.cmd_timer_callback()
        command = outputs[-1] if outputs else None
        drive.vx = command.linear.x if command else 0.
        drive.wz = command.angular.z if command else 0.
        if tick%30 == 0:
            print("pose",round(data.time,2),runtime.base_pos().round(3),round(drive.vx,3),round(drive.wz,3),flush=True)
        if renderer:
            record_frame()
        for _ in range(5):
            runtime.tick()
            assert not runtime.is_fallen(), ("fallen",runtime.base_pos())
            for contact in data.contact[:data.ncon]:
                assert not wall_ids.intersection((contact.geom1, contact.geom2)), ("wall contact",runtime.base_pos())
            q = data.qpos[binding.root_qpos_adr:binding.root_qpos_adr+7]
            clearance = course.shaft_clearance(q[0],q[1],course.yaw_of(q[3:]), course.MULTI_SHAFT if flights > 2 else course.SHAFT)
            min_clearance=min(min_clearance,clearance)
            assert clearance>0, ("shaft",q[:3])
        if stair.arrived:
            assert stair.gen.landings==flights and drive.vx==drive.wz==0
            stop = runtime.base_pos().copy()
            assert stop[0] < -.5 and abs(stop[2] + 1.44 * flights - robot.spawn_drop) < .12, stop
            for i in range(75):
                stair.timer_callback()
                control.cmd_timer_callback()
                assert stair.arrived and stair.gen.landings == flights
                assert outputs[-1].linear.x == outputs[-1].angular.z == 0
                runtime.tick()
                assert not runtime.is_fallen(), ("fallen at stop",runtime.base_pos())
                if renderer and i % 5 == 4:
                    record_frame()
            hold = runtime.base_pos().copy()
            assert np.linalg.norm(hold[:2] - stop[:2]) < .12
            print(f"PASS DOWN {floors} floors", stop, "hold", hold, "shaft clearance", min_clearance, flush=True)
            if capture_path:
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(capture_path, detail=frames, time=capture_times,
                                    qpos=capture_poses, bev=capture_maps, target=capture_targets,
                                    path=capture_paths, cmd=capture_commands, status=capture_status,
                                    landings=capture_landings, clearance=capture_clearance,
                                    stop=stop, hold=hold, landing_events=landing_events, floors=floors)
            break
    else:
        raise RuntimeError(("did not arrive",runtime.base_pos(),stair.gen.landings))
finally:
    if capture_path and frames and not stair.arrived:
        np.savez_compressed(capture_path, detail=frames, time=capture_times, qpos=capture_poses, bev=capture_maps, target=capture_targets, path=capture_paths, cmd=capture_commands, status=capture_status, landings=capture_landings, clearance=capture_clearance)
    for n in (stair, planner,control):
        n.destroy_node()
    rclpy.shutdown()
    if renderer:
        renderer.close()
    nav_camera.close()
    runtime.close()

"""Exercise finite UP/DOWN floors through stair_node, planning_node, and cmd_vel_control.

Requires the Go2 MuJoCo runtime and the stair simulation course builder.
All ROS publishers stay on localhost in the test container.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from std_msgs.msg import Bool

from tool.stair_odom_noise import OdomNoise, OdomNoiseConfig
from tool.stair_sim_control import apply_drive_command

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sim-root", type=Path, required=True)
parser.add_argument("--course-script", type=Path, default=Path(__file__).with_name("stair_sim_course.py"))
parser.add_argument("--capture", type=Path)
parser.add_argument("--direction", choices=('up', 'down'), default='down')
parser.add_argument("--floors", type=int, choices=range(1, 51), default=1)
parser.add_argument("--levels", type=int, choices=range(2, 52),
                    help="Number of full-floor platforms; default: task floors + 1")
parser.add_argument("--flight-width", type=float, help="Clear stair flight width in meters")
parser.add_argument("--landing-depth", type=float, default=1.75)
parser.add_argument('--open-landing', action='append', type=int, default=[], metavar='FLIGHT',
                    help='Open an outward room doorway on a zero-based landing; repeatable')
parser.add_argument('--odom-noise', choices=('none', 'position', 'yaw', 'pitch', 'combined'), default='none')
parser.add_argument('--noise-seed', type=int, default=7)
parser.add_argument('--noise-scale', type=float, default=1.)
parser.add_argument('--noise-position-std', type=float, default=.01, help='Per-world-axis position standard deviation, meters')
parser.add_argument('--noise-yaw-std', type=float, default=.5, help='World-Z yaw standard deviation, degrees')
parser.add_argument('--noise-pitch-std', type=float, default=.5, help='Optical-X pitch standard deviation, degrees')
parser.add_argument('--noise-pitch-amplitude', type=float, default=1., help='Periodic optical-X pitch peak amplitude, degrees')
parser.add_argument('--noise-pitch-hz', type=float, default=1.7)
parser.add_argument('--noise-correlation', type=float, default=.2, help='OU correlation time, seconds')
parser.add_argument('--max-sim-seconds', type=float, help='Optional stricter simulation time budget; does not extend the existing budget')
parser.add_argument('--result-json', type=Path, help='Write outcome and odom/target/command traces even on assertion failure')
args = parser.parse_args()
if args.max_sim_seconds is not None and (not np.isfinite(args.max_sim_seconds) or args.max_sim_seconds <= 0):
    parser.error('--max-sim-seconds must be finite and positive')
max_sim_seconds = args.max_sim_seconds
capture_path = args.capture
result_path = args.result_json
noise_config = OdomNoiseConfig(mode=args.odom_noise, seed=args.noise_seed, scale=args.noise_scale,
    position_std_m=args.noise_position_std, yaw_std_deg=args.noise_yaw_std,
    pitch_std_deg=args.noise_pitch_std, pitch_amplitude_deg=args.noise_pitch_amplitude,
    pitch_hz=args.noise_pitch_hz, correlation_s=args.noise_correlation)
floors = args.floors
direction = args.direction
rise_sign = 1 if direction == 'up' else -1
flights = floors * 2
levels = args.levels or floors + 1
if levels <= floors:
    parser.error("--levels must be greater than the relative --floors task")
scene_flights = 2 * (levels - 1)
if any(i < 0 or i >= scene_flights for i in args.open_landing):
    parser.error('--open-landing must identify a flight in this scene')
open_landings = sorted(set(args.open_landing))
max_ticks = max(1800, flights * 900)
if max_sim_seconds is not None:
    max_ticks = min(max_ticks, int(np.ceil(max_sim_seconds / .1)))
noise = OdomNoise(noise_config, max_ticks * .1 + 5.)
noise_json = json.dumps(noise.metadata(), sort_keys=True)
print('ODOM_NOISE', noise_json, flush=True)
lane_width, landing_depth = args.flight_width, args.landing_depth
if direction == 'up' and lane_width is None:
    # The legacy two-flight course is descending-only; use the signed stack.
    lane_width = 1.8
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
model, binding, surfaces, walls = course.build_world(
    plant, robot, scene_flights, direction=rise_sign, mapping_rig=False,
    lane_width=lane_width, landing_depth=landing_depth, doors={i: 'room' for i in open_landings})
shaft = course.shaft_bounds(scene_flights, lane_width, landing_depth)
print(f"SCENE direction={direction} levels={levels} flights={scene_flights} flight_width={lane_width} landing_depth={landing_depth} task_floors={floors}", flush=True)
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
args = SimpleNamespace(camera_height=.4669, direction=direction, floors=floors,
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
            if model.geom(i).name.endswith(("_wall", "door_leaf")) or "rail" in model.geom(i).name}
min_opening_clearance = np.inf
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
validated = False
failure = None
trace_time, true_poses, estimated_poses, noise_errors = [], [], [], []
trace_target, trace_cmd, trace_status, trace_counts = [], [], [], []
stop = hold = None
try:
    for tick in range(max_ticks):
        mujoco.mj_camlight(model,data)
        camid = runtime.b.depth_camera_id
        T = np.eye(4)
        T[:3,:3] = data.cam_xmat[camid].reshape(3,3) @ np.diag((1.,-1.,-1.))
        T[:3,3] = data.cam_xpos[camid]
        elapsed = float(data.time) - start_time
        estimated = noise.apply(T, elapsed)
        trace_time.append(elapsed)
        true_poses.append(T.copy())
        estimated_poses.append(estimated.copy())
        noise_errors.append(noise.error(elapsed))
        trace_target.append(np.full(2, np.nan) if planner.target_pose is None else planner.target_pose[:2].copy())
        trace_cmd.append((drive.vx, drive.wz))
        trace_status.append(status.last or '')
        trace_counts.append(stair.gen.landings)
        pose = np2msg(estimated, stair.get_clock().now().to_msg(), "world", "camera")
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
            assert abs(base[2] - rise_sign * 1.44 * last_landing - robot.spawn_drop) < .18, ("count height", last_landing, base)
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
        trace_target[-1] = np.full(2, np.nan) if planner.target_pose is None else planner.target_pose[:2].copy()
        trace_cmd[-1] = (drive.vx, drive.wz)
        trace_status[-1] = status.last or ''
        trace_counts[-1] = stair.gen.landings
        if tick%30 == 0:
            print("pose",round(data.time,2),runtime.base_pos().round(3),round(drive.vx,3),round(drive.wz,3),flush=True)
        if renderer:
            record_frame()
        for _ in range(5):
            apply_drive_command(drive, outputs[-1] if outputs else None)
            runtime.tick()
            assert not runtime.is_fallen(), ("fallen",runtime.base_pos())
            for contact in data.contact[:data.ncon]:
                assert not wall_ids.intersection((contact.geom1, contact.geom2)), ("wall contact",runtime.base_pos())
            q = data.qpos[binding.root_qpos_adr:binding.root_qpos_adr+7]
            clearance = course.shaft_clearance(q[0],q[1],course.yaw_of(q[3:]), shaft)
            min_clearance=min(min_clearance,clearance)
            assert clearance>0, ("shaft",q[:3])
            for flight in open_landings:
                landing_z = rise_sign * 1.44 * (flight + 1) + robot.spawn_drop
                if abs(q[2] - landing_z) < .4:
                    margin = course.landing_opening_clearance(q[0], course.yaw_of(q[3:]), flight, landing_depth)
                    min_opening_clearance = min(min_opening_clearance, margin)
                    assert margin > 0, ('landing opening', flight, q[:3])
        if stair.arrived:
            assert stair.gen.landings==flights and drive.vx==drive.wz==0
            stop = runtime.base_pos().copy()
            assert stop[0] < -.5 and abs(stop[2] - rise_sign * 1.44 * flights - robot.spawn_drop) < .12, stop
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
            validated = True
            print(f"PASS {direction.upper()} {floors} floors", stop, "hold", hold, "shaft clearance", min_clearance, flush=True)
            if capture_path:
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(capture_path, detail=frames, time=capture_times,
                                    qpos=capture_poses, bev=capture_maps, target=capture_targets,
                                    path=capture_paths, cmd=capture_commands, status=capture_status,
                                    landings=capture_landings, clearance=capture_clearance,
                                    stop=stop, hold=hold, landing_events=landing_events, floors=floors,
                                    levels=levels, flight_width=np.nan if lane_width is None else lane_width,
                                    landing_depth=landing_depth, shaft=shaft, arrived=True,
                                    min_clearance=min_clearance, direction=direction, odom_noise=noise_json)
            break
    else:
        raise RuntimeError(("did not arrive",runtime.base_pos(),stair.gen.landings))
except (Exception, KeyboardInterrupt) as exc:
    failure = f'{type(exc).__name__}: {exc}'
    raise
finally:
    if result_path:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path = result_path.with_suffix('.trace.npz')
        np.savez_compressed(trace_path, time=trace_time, true_pose=true_poses,
                            estimated_pose=estimated_poses, error=noise_errors,
                            target=trace_target, cmd=trace_cmd, status=trace_status,
                            landings=trace_counts, noise=noise_json)
        result_path.write_text(json.dumps(dict(passed=validated, failure=failure,
            direction=direction, floors=floors, levels=levels, flight_width=lane_width,
            time_budget_s=max_ticks * .1,
            landing_depth=landing_depth, noise=noise.metadata(),
            open_landings=open_landings,
            min_opening_clearance_m=float(min_opening_clearance) if np.isfinite(min_opening_clearance) else None,
            simulated_s=float(data.time) - start_time, landings=int(stair.gen.landings),
            node_arrived=bool(stair.arrived), min_shaft_clearance_m=float(min_clearance),
            stop=None if stop is None else stop.tolist(), hold=None if hold is None else hold.tolist(),
            landing_events=landing_events, trace_file=trace_path.name), indent=2))
    if capture_path and frames and not validated:
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(capture_path, detail=frames, time=capture_times,
                            qpos=capture_poses, bev=capture_maps, target=capture_targets,
                            path=capture_paths, cmd=capture_commands, status=capture_status,
                            landings=capture_landings, clearance=capture_clearance,
                            floors=floors, levels=levels, landing_depth=landing_depth,
                            flight_width=np.nan if lane_width is None else lane_width,
                            shaft=shaft, arrived=False, node_arrived=stair.arrived,
                            min_clearance=min_clearance, direction=direction, odom_noise=noise_json)
    for n in (stair, planner,control):
        n.destroy_node()
    rclpy.shutdown()
    if renderer:
        renderer.close()
    nav_camera.close()
    runtime.close()

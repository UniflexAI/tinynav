"""Interactive browser host for the stair-floors development simulation."""
import contextlib
import copy
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(os.environ.get('TINYNAV_SIM_ROOT', '/simrepo'))
OUTPUT = Path(os.environ.get('TINYNAV_SIM_OUTPUT', '/simwork/output/live'))
OUTPUT.mkdir(parents=True, exist_ok=True)
EMBEDDED = os.environ.get('TINYNAV_SIM_EMBEDDED') == '1'
sys.path[:0] = [str(ROOT), str(SIM_ROOT / 'mujoco')]
import cv2
import mujoco
import numpy as np
import rclpy
import viser
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from sim import plant
from sim.keyboard import ConstantCmd
from sim.policy import PieOnnxPolicy
from sim.robots import get_robot
from sim.runtime import PieRuntime, DepthCamera
from webviewer import ViserMujocoScene
from tinynav.core import stair_node, planning_node
from tinynav.core.math_utils import np2msg
from tinynav.platforms import cmd_vel_control
from stair_bev_view import render_bev
from tool.stair_odom_noise import OdomNoise, OdomNoiseConfig
from tool.stair_sim_control import apply_drive_command
from tool.stair_app_bridge import StairAppBridge
from tool.stair_sim_api import SimulationControl

spec = importlib.util.spec_from_file_location('course', str(ROOT / 'tool/stair_sim_course.py'))
course = importlib.util.module_from_spec(spec)
spec.loader.exec_module(course)
server = viser.ViserServer(host='0.0.0.0', port=int(os.environ.get('TINYNAV_SIM_VIEWER_PORT', '8012')))
api = SimulationControl()
if os.environ.get('TINYNAV_SIM_CONTROL_PORT'):
    api.serve(int(os.environ['TINYNAV_SIM_CONTROL_PORT']))
server.gui.configure_theme(control_layout='floating', control_width='medium')
intro = server.gui.add_markdown('## TinyNav stair development\nMuJoCo + Go2 policy + current stair/planner/controller code.\nMouse: orbit / zoom / pan. Start begins a new task in the selected direction.')
scene_choice = server.gui.add_dropdown('Scene', options=('Baseline wide', 'Ten-level narrow'), initial_value='Baseline wide')
scene_note = server.gui.add_markdown('Baseline: original wide course, DOWN 1/2 validated.')
direction_choice = server.gui.add_dropdown('Direction', options=('down', 'up'), initial_value='down')
floors = server.gui.add_dropdown('Floors to traverse', options=('1', '2'), initial_value='2')
landing_depth_choice = server.gui.add_dropdown('Narrow landing depth (m)', options=('1.75', '1.2'), initial_value='1.75')
landing_depth_choice.disabled = True
@scene_choice.on_update
def select_scene(_):
    narrow = scene_choice.value != 'Baseline wide'
    floors.options = tuple(str(n) for n in range(1,10)) if narrow else ('1', '2')
    landing_depth_choice.disabled = not narrow
    if floors.value not in floors.options:
        floors.value = '2'
    scene_note.content = ('10 full-floor platforms, 18 flights, 1.2 m clear width, Selectable 1.75 / 1.2 m landing depth. Both depths passed UP 9 and DOWN 9 isolated simulation regressions with 18 landings and a zero-command hold. Safety assertions stay enabled. This is not physical-robot validation. Press Start to apply.' if narrow else 'Baseline: original wide course, DOWN 1/2 validated. Press Start to apply.')
noise_mode = server.gui.add_dropdown('Odom noise', options=('none', 'position', 'yaw', 'pitch', 'combined'), initial_value='none')
noise_scale = server.gui.add_dropdown('Noise scale', options=('0.5', '1', '2'), initial_value='0.5')
noise_seed = server.gui.add_dropdown('Noise seed', options=('7', '17', '27'), initial_value='7')
noise_help = server.gui.add_markdown('Noise: none=off; position / yaw / pitch / combined. Scale 1: XYZ std 1 cm/axis, yaw std 0.5 deg, pitch std 0.5 deg + 1 deg peak at 1.7 Hz. All noise amplitudes scale together. Apply settings with Start / Reset. Noise affects navigation poses only; independent safety checks use truth. Single-seed experiments are not a real VIO tolerance.')
noise_text = server.gui.add_markdown('Applied odom noise: none')
noise_image = server.gui.add_image(np.zeros((200,480,3), dtype=np.uint8), label='Odom error history (10 s)')
recording_view = server.gui.add_checkbox('Recording view', initial_value=False, order=-1)
sim_speed = server.gui.add_dropdown('Simulation speed', options=('1', '2', '4'), initial_value='1')
with server.gui.add_folder('Scene controls', expand_by_default=False) if EMBEDDED else contextlib.nullcontext():
    camera_mode = server.gui.add_dropdown('Third-person camera', options=('Free orbit', 'Follow robot', 'Overview'), initial_value='Free orbit')
    reset_environment = server.gui.add_button('Reset environment (paused)', visible=EMBEDDED)

@camera_mode.on_update
def update_camera_mode(_):
    current = api.snapshot()
    if camera_mode.value == 'Overview' and current.get('ready'):
        base = np.array(current['position'])
        for client in server.get_clients().values():
            client.camera.position = base + (6., -7., 5.)
            client.camera.look_at = base
            client.camera.up_direction = (0., 0., 1.)

@reset_environment.on_click
def reset_from_viewer(_):
    current = api.snapshot()
    if not current.get('ready'):
        return
    cfg = dict(direction=current['direction'], floors=current['floors'],
               landing_depth=current['landing_depth'], speed=current['speed'],
               noise_mode=current['odom_noise']['mode'],
               noise_scale=current['odom_noise']['scale'], noise_seed=current['odom_noise']['seed'])
    api.submit('reset', cfg)
start = server.gui.add_button('Start / Reset')
pause = server.gui.add_checkbox('Pause physics', initial_value=True)
manual = server.gui.add_checkbox('Manual drive', initial_value=False)
vx = server.gui.add_slider('Forward m/s', min=-0.5, max=0.5, step=0.05, initial_value=0.0)
wz = server.gui.add_slider('Turn rad/s', min=-1.0, max=1.0, step=0.1, initial_value=0.0)
stop = server.gui.add_button('STOP')
presentation_summary = server.gui.add_markdown('MuJoCo simulation', visible=False)
state_text = server.gui.add_markdown('Ready')
bev_planner_image = server.gui.add_image(np.zeros((502,400,3),dtype=np.uint8), label='TinyNav planner BEV (live)')
bev_height_image = server.gui.add_image(np.zeros((502,400,3),dtype=np.uint8), label='Stair height BEV (live)')
depth_image = server.gui.add_image(np.zeros((160,136,3),dtype=np.uint8), label='Front measured depth')
@recording_view.on_update
def set_recording_view(_):
    compact = recording_view.value
    for handle in (intro, scene_choice, scene_note, direction_choice, floors,
                   landing_depth_choice, noise_mode, noise_scale, noise_seed,
                   noise_help, noise_text, noise_image, sim_speed, manual,
                   vx, wz, stop, bev_height_image):
        handle.visible = not compact
    presentation_summary.visible = compact

request = {'reset': True, 'run': False}

@start.on_click
def on_start(_):
    request.update(reset=True, run=True)

@stop.on_click
def on_stop(_):
    pause.value = True
    vx.value = wz.value = 0.0

if EMBEDDED:
    scene_choice.value = 'Ten-level narrow'
    floors.options = tuple(str(n) for n in range(1,10))
    floors.value = '9'
    landing_depth_choice.value = '1.2'
    noise_mode.value = 'combined'
    camera_mode.value = 'Follow robot'
    for handle in (intro, scene_choice, scene_note, direction_choice, floors,
                   landing_depth_choice, noise_mode, noise_scale, noise_seed,
                   noise_help, noise_text, noise_image, sim_speed, manual,
                   vx, wz, stop, start, pause, recording_view, bev_height_image):
        handle.visible = False
    presentation_summary.visible = True

rclpy.init()
world = None

def publish_app_color(w):
    w.rgb_renderer.update_scene(w.data, camera=w.binding.depth_camera_id, scene_option=w.rgb_options)
    rgb = w.rgb_renderer.render()
    api.image('color-high', cv2.imencode('.jpg', rgb[...,::-1], [cv2.IMWRITE_JPEG_QUALITY,80])[1])
    api.image('color', cv2.imencode('.jpg', cv2.resize(rgb,(272,320))[...,::-1], [cv2.IMWRITE_JPEG_QUALITY,50])[1])


def create_world(n):
    global world
    if world:
        if not world.trace_written:
            save_live_trace(world)
        for node in (world.stair, world.planner, world.control):
            node.destroy_node()
        world.camera.close()
        world.rgb_renderer.close()
        world.runtime.close()
    server.scene.reset()
    robot = get_robot('go2')
    narrow=scene_choice.value != 'Baseline wide'
    levels=10 if narrow else n+1
    scene_flights=2*(levels-1)
    lane_width=1.2 if narrow else None
    landing_depth=float(landing_depth_choice.value) if narrow else 1.75
    direction=direction_choice.value
    rise_sign=1 if direction=='up' else -1
    if direction=='up' and lane_width is None:
        lane_width=1.8
    shaft=course.shaft_bounds(scene_flights,lane_width,landing_depth)
    model,binding,_,_ = course.build_world(plant,robot,scene_flights,direction=rise_sign,mapping_rig=False,lane_width=lane_width,landing_depth=landing_depth)
    data=mujoco.MjData(model)
    plant.reset_to_spawn(model,data,binding,pos=course.SPAWN,yaw_deg=7.)
    drive=ConstantCmd(0.,0.)
    runtime=PieRuntime(model,binding,data,PieOnnxPolicy(Path(robot.policy_path('perception')),provider='cpu'),None,drive)
    for _ in range(50): runtime.tick()
    course.FrontCameraBEV(runtime, profile='ros-left')
    camera=DepthCamera(model,binding.depth_camera_id,640,544)
    rgb_renderer=mujoco.Renderer(model,height=640,width=544)
    rgb_options=mujoco.MjvOption()
    rgb_options.geomgroup[:] = 0
    rgb_options.geomgroup[:3] = 1
    stair=stair_node.StairNode(SimpleNamespace(camera_height=.4669,direction=direction,floors=n,landings_per_floor=2,turn='auto',memory=None,rate=2.))
    planner=planning_node.PlanningNode()
    control=cmd_vel_control.CmdVelControlNode()
    clock=SimpleNamespace(monotonic=lambda:float(data.time))
    for module in (stair_node,planning_node,cmd_vel_control): module.time=clock
    stair.K=planner.K=np.array([[303.10,0.,269.37],[0.,303.10,314.67],[0.,0.,1.]])
    control._on_nav_active(Bool(data=True))
    w=SimpleNamespace(model=model,data=data,binding=binding,drive=drive,runtime=runtime,camera=camera,stair=stair,planner=planner,control=control,tick=0,status='Ready',outputs=[],min_clearance=float('inf'),floors=n,bev_result={},bev_planner=None,bev_planner_origin=None,bev_mask=None,bev_path=[],bev_cam_z=0.,bev_frames=0)
    w.rgb_renderer, w.rgb_options = rgb_renderer, rgb_options
    w.app_bridge = StairAppBridge()
    w.levels,w.scene_flights,w.lane_width,w.landing_depth,w.shaft=levels,scene_flights,lane_width,landing_depth,shaft
    w.direction=direction
    w.scene_name=scene_choice.value
    w.error=''
    w.cancelled=False
    w.teleop_until=0.
    w.manual_vy=0.
    w.noise = OdomNoise(OdomNoiseConfig(mode=noise_mode.value, scale=float(noise_scale.value), seed=int(noise_seed.value)), 3600.)
    w.noise_start = float(data.time)
    w.was_paused = True
    w.resume_events = []
    w.trace = {key: [] for key in ('time','true_pose','estimated_pose','error','target','cmd','status','landings')}
    w.trace_written = False
    w.run_id = f'{direction}-{n}-{time.time_ns()}'
    w.arrival_stop = None
    w.hold_started = None
    w.validated = False
    w.last_landing = 0
    w.landing_events = []
    w.robot = robot
    w.noise_error = np.zeros(5)
    w.noise_history = []
    w.estimated_position = None
    noise_image.image = np.zeros((200,480,3), dtype=np.uint8)
    noise_text.content = f'Applied: **{w.noise.config.mode}**, scale={w.noise.config.scale:g}, seed={w.noise.config.seed}. Waiting for poses.'
    original_compute=stair.gen.compute
    def capture_compute(*args,**kwargs):
        result=original_compute(*args,**kwargs)
        w.bev_result=result
        w.bev_cam_z=float(args[0][2,3])
        return result
    stair.gen.compute=capture_compute
    height_publisher=planner.height_map_pub
    def capture_height(msg):
        w.bev_planner=planner.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')[...,::-1].copy()
        w.bev_planner_origin=planner.origin.copy()
        height_publisher.publish(msg)
        w.app_bridge._on_height_map(msg)
    planner.height_map_pub=SimpleNamespace(publish=capture_height)
    obstacle_publisher=planner.obstacle_mask_pub
    def capture_obstacles(msg):
        w.bev_mask=msg
        obstacle_publisher.publish(msg)
        w.app_bridge._on_obstacle_mask(msg)
    planner.obstacle_mask_pub=SimpleNamespace(publish=capture_obstacles)
    def status(msg):
        w.status=msg.data
        planner._on_stair_status(msg)
        control._on_stair_status(msg)
    stair.status_pub=SimpleNamespace(publish=status)
    def capture_target(msg):
        planner.target_pose_callback(msg)
        w.app_bridge._on_nav_target_pose(msg)
    stair.target_pub=SimpleNamespace(publish=capture_target)
    stair.poi_change_pub=SimpleNamespace(publish=planner.poi_change_callback)
    def capture_path(msg):
        w.bev_path=[(p.pose.position.x,p.pose.position.y) for p in msg.poses]
        control.path_callback(msg)
        w.app_bridge._on_trajectory_path(msg)
    planner.path_pub=SimpleNamespace(publish=capture_path)
    footprint_publisher=planner.footprint_pub
    def capture_footprint(msg):
        footprint_publisher.publish(msg)
        w.app_bridge._on_footprint(msg)
    planner.footprint_pub=SimpleNamespace(publish=capture_footprint)
    control.cmd_pub=SimpleNamespace(publish=lambda msg:w.outputs.__setitem__(slice(None),[msg]))
    view_model=copy.copy(model)
    if recording_view.value or EMBEDDED:
        original_rgba = model.geom_rgba.copy()
        for gid in range(view_model.ngeom):
            name = view_model.geom(gid).name
            if name.endswith('_wall') or 'rail' in name:
                view_model.geom_rgba[gid,3] = .12
        assert np.array_equal(model.geom_rgba, original_rgba)
    view_model.geom_group[view_model.geom_group==3]=2
    w.scene=ViserMujocoScene(server,view_model,num_envs=1)
    w.scene.camera_tracking_enabled=False
    w.scene.update_from_mjdata(data)
    height_bev,planner_bev,_=render_bev(w)
    bev_height_image.image=height_bev
    bev_planner_image.image=np.pad(planner_bev[:434], ((0,0),(40,40),(0,0))) if EMBEDDED else planner_bev[:434] if recording_view.value else planner_bev
    mujoco.mj_camlight(model, data)
    publish_app_color(w)
    initial_T=np.eye(4)
    initial_T[:3,:3]=data.cam_xmat[binding.depth_camera_id].reshape(3,3)@np.diag((1.,-1.,-1.))
    initial_T[:3,3]=data.cam_xpos[binding.depth_camera_id]
    initial_pose=np2msg(initial_T,stair.get_clock().now().to_msg(),'world','camera')
    initial_pose.header.stamp.sec=int(data.time)
    initial_pose.header.stamp.nanosec=int(data.time%1*1e9)
    w.app_bridge._on_slam_odom(initial_pose)
    api.publish_planning(w.app_bridge.snapshot())
    preview = camera.capture(data)
    preview = cv2.applyColorMap((np.clip(preview, 0, 5)/5*255).astype(np.uint8), cv2.COLORMAP_TURBO)[...,::-1]
    api.image('depth-high', cv2.imencode('.jpg', preview[...,::-1], [cv2.IMWRITE_JPEG_QUALITY,80])[1])
    preview = cv2.resize(preview, (272, 320))
    if EMBEDDED:
        preview_panel = np.zeros((320,500,3), dtype=np.uint8)
        preview_panel[:,114:386] = preview
        depth_image.image = preview_panel
    else:
        depth_image.image = preview
    api.image('depth', cv2.imencode('.jpg', preview[...,::-1])[1])
    api.image('planner', cv2.imencode('.jpg', planner_bev[...,::-1])[1])
    for client in server.get_clients().values():
        client.camera.position=(6.,-7.,5.)
        client.camera.look_at=(1.5,.5,-1.5)
        client.camera.up_direction=(0.,0.,1.)
    return w

@server.on_client_connect
def on_connect(client):
    client.camera.position=(6.,-7.,5.)
    client.camera.look_at=(1.5,.5,-1.5)
    client.camera.up_direction=(0.,0.,1.)

def save_live_trace(w):
    trace_dir = OUTPUT / 'runs'
    trace_dir.mkdir(exist_ok=True)
    np.savez_compressed(trace_dir / (w.run_id+'.npz'), **w.trace)
    (trace_dir / (w.run_id+'.json')).write_text(json.dumps(dict(direction=w.direction,
        floors=w.floors,noise=w.noise.metadata(),validated=w.validated,error=w.error,
        cancelled=w.cancelled,flight_width=w.lane_width,landing_depth=w.landing_depth,
        stop=None if w.arrival_stop is None else w.arrival_stop.tolist(),
        final=w.runtime.base_pos().tolist(),min_clearance=w.min_clearance,
        landings=w.stair.gen.landings,landing_events=w.landing_events,
        resume_events=w.resume_events,simulated_s=float(w.data.time)-w.noise_start),indent=2))
    w.trace_written = True

def apply_api_command(command):
    global world
    action, values = command.action, command.payload
    if action in ('start', 'reset'):
        scene_choice.value = 'Ten-level narrow'
        floors.options = tuple(str(n) for n in range(1,10))
        direction_choice.value = values['direction']
        floors.value = str(values['floors'])
        landing_depth_choice.value = str(values['landing_depth'])
        noise_mode.value = values['noise_mode']
        noise_scale.value = str(values['noise_scale']).removesuffix('.0')
        noise_seed.value = str(values['noise_seed'])
        sim_speed.value = str(values['speed'])
        pause.value = True
        with api.lock:
            api.images.clear()
        world = create_world(values['floors'])
        manual.value = False
        vx.value = wz.value = 0.
        pause.value = action == 'reset'
        request['reset'] = False
    elif action == 'teleop':
        if world.error:
            raise ValueError('Safety assertion stopped simulation; reset first')
        moving=any(values[k] for k in ('linear_x','linear_y','angular_z'))
        if moving and not manual.value:
            if not world.trace_written:
                save_live_trace(world)
            world.cancelled=True
            world.validated=False
            world.control._on_nav_active(Bool(data=False))
            world.outputs.clear()
            manual.value=True
        if manual.value:
            vx.value=values['linear_x']; wz.value=values['angular_z']
            world.manual_vy=values['linear_y']
            world.teleop_until=time.monotonic()+.5
            if moving: pause.value=False
            world.status='manual drive'
    elif action == 'camera':
        recording_view.value = False
        camera_mode.value = values['mode']
        if values['mode'] == 'Overview':
            base = world.runtime.base_pos()
            for client in server.get_clients().values():
                client.camera.position = base + (6., -7., 5.)
                client.camera.look_at = base
                client.camera.up_direction = (0., 0., 1.)
    elif action == 'resume':
        if world.error or world.validated or world.cancelled:
            raise ValueError('Task ended; start a new task or reset')
        pause.value = False
    else:
        pause.value = True
        if action == 'stop':
            world.cancelled = True
            world.outputs.clear()
            world.control._on_nav_active(Bool(data=False))


bridge=CvBridge()
print('LIVE_SIM_READY port=8012',flush=True)
while True:
    begin=time.perf_counter()
    api_command = api.next_command()
    if api_command:
        try:
            apply_api_command(api_command)
        except Exception as exc:
            pause.value = True
            api_command.error = str(exc)
            api_command.done.set()
            api_command = None
    if request['reset']:
        request['reset']=False
        world=create_world(int(floors.value))
        manual.value=False
        vx.value=wz.value=0.
        pause.value=not request['run']
    w=world
    if w.error:
        pause.value=True
    try:
        if not pause.value:
            if w.tick%5==0:
                mujoco.mj_camlight(w.model,w.data)
                camid=w.binding.depth_camera_id
                T=np.eye(4)
                T[:3,:3]=w.data.cam_xmat[camid].reshape(3,3)@np.diag((1.,-1.,-1.))
                T[:3,3]=w.data.cam_xpos[camid]
                elapsed = float(w.data.time) - w.noise_start
                estimated = w.noise.apply(T, elapsed)
                w.noise_error = w.noise.error(elapsed)
                w.estimated_position = estimated[:3,3].copy()
                w.noise_history.append(w.noise_error.copy())
                w.noise_history = w.noise_history[-100:]
                pose=np2msg(estimated,w.stair.get_clock().now().to_msg(),'world','camera')
                pose.header.stamp.sec=int(w.data.time)
                pose.header.stamp.nanosec=int(w.data.time%1*1e9)
                w.app_bridge._on_slam_odom(pose)
                raw=w.camera.capture(w.data)
                depth=bridge.cv2_to_imgmsg(raw,encoding='32FC1'); depth.header=pose.header
                if not manual.value:
                    w.stair.odom_callback(pose); w.stair.depth_callback(depth,pose)
                if w.stair.gen.landings != w.last_landing:
                    assert w.stair.gen.landings == w.last_landing + 1, 'Skipped landing count'
                    w.last_landing = w.stair.gen.landings
                    base = w.runtime.base_pos().copy()
                    sign = 1 if w.direction == 'up' else -1
                    assert abs(base[2] - sign*1.44*w.last_landing - w.robot.spawn_drop) < .18, ('count height',w.last_landing,base)
                    w.landing_events.append((float(w.data.time)-w.noise_start,w.last_landing,*base))
                w.control.pose_callback(pose)
                if w.tick%25==0 and not manual.value: w.stair.timer_callback()
                with contextlib.redirect_stdout(io.StringIO()): w.planner.sync_callback(depth,pose)
                if w.tick%10==0:
                    e = w.noise_error
                    noise_text.content = (f'Applied: **{w.noise.config.mode}**, scale={w.noise.config.scale:g}, seed={w.noise.config.seed}\n\n'
                        f'XYZ error: {e[0]*1000:+.1f}, {e[1]*1000:+.1f}, {e[2]*1000:+.1f} mm | yaw={e[3]:+.2f} deg, pitch={e[4]:+.2f} deg')
                    graph = np.full((200,480,3), (16,24,35), dtype=np.uint8)
                    history = np.array(w.noise_history)
                    colors = [(255,130,120),(100,225,160),(120,175,255),(255,210,100),(210,145,255)]
                    pos_limit = max(20., 40*w.noise.config.scale)
                    angle_limit = max(2., 4*w.noise.config.scale)
                    for center in (60,150): cv2.line(graph,(0,center),(479,center),(70,82,100),1)
                    for j,color in enumerate(colors):
                        values = history[:,j] * (1000 if j<3 else 1)
                        center, limit = (60,pos_limit) if j<3 else (150,angle_limit)
                        points = np.column_stack((np.arange(100-len(values),100)*479/99,center-np.clip(values/limit,-1,1)*33)).astype(np.int32)
                        if len(points)>1: cv2.polylines(graph,[points],False,color,1,cv2.LINE_AA)
                    cv2.putText(graph,f'XYZ mm (+/-{pos_limit:g})',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,230,245),1,cv2.LINE_AA)
                    cv2.putText(graph,f'yaw / pitch deg (+/-{angle_limit:g})',(8,107),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,230,245),1,cv2.LINE_AA)
                    noise_image.image = graph
                    height_bev,planner_bev,observed=render_bev(w)
                    bev_height_image.image=height_bev
                    bev_planner_image.image=np.pad(planner_bev[:434], ((0,0),(40,40),(0,0))) if EMBEDDED else planner_bev[:434] if recording_view.value else planner_bev
                    w.bev_frames+=1
                    w.app_bridge.capture_voxels(w.planner)
                    api.publish_planning(w.app_bridge.snapshot())
                    publish_app_color(w)
                    api.image('planner', cv2.imencode('.jpg', planner_bev[...,::-1])[1])
                    if w.bev_frames%5==1:
                        cv2.imwrite(str(OUTPUT / 'bev-height.png'),height_bev[...,::-1])
                        cv2.imwrite(str(OUTPUT / 'bev-planner.png'),planner_bev[...,::-1])
                        Path(OUTPUT / 'bev-state.json').write_text(json.dumps({'frames':w.bev_frames,'sim_time':float(w.data.time),'observed_cells':observed,'path_points':len(w.bev_path),'target':None if w.planner.target_pose is None else w.planner.target_pose.tolist()}))
                w.control.cmd_timer_callback()
                cmd=w.outputs[-1] if w.outputs else None
                w.trace['time'].append(elapsed)
                w.trace['true_pose'].append(T.copy())
                w.trace['estimated_pose'].append(estimated.copy())
                w.trace['error'].append(w.noise_error.copy())
                w.trace['target'].append(np.full(2,np.nan) if w.planner.target_pose is None else w.planner.target_pose[:2].copy())
                w.trace['cmd'].append((cmd.linear.x,cmd.angular.z) if cmd else (0.,0.))
                w.trace['status'].append(w.status)
                w.trace['landings'].append(w.stair.gen.landings)
                pixels=(np.clip(raw,0,5)/5*255).astype(np.uint8)
                if w.tick%10==0:
                    full_depth=cv2.applyColorMap(pixels,cv2.COLORMAP_TURBO)
                    api.image('depth-high',cv2.imencode('.jpg',full_depth,[cv2.IMWRITE_JPEG_QUALITY,80])[1])
                colored_depth = cv2.applyColorMap(cv2.resize(pixels,(136,160)),cv2.COLORMAP_TURBO)[...,::-1]
                api.image('depth', cv2.imencode('.jpg', colored_depth[...,::-1])[1])
                if recording_view.value or EMBEDDED:
                    depth_panel = np.zeros((320,500 if EMBEDDED else 400,3),dtype=np.uint8)
                    left = 114 if EMBEDDED else 64
                    depth_panel[:,left:left+272] = cv2.resize(colored_depth,(272,320),interpolation=cv2.INTER_NEAREST)
                    depth_image.image = depth_panel
                else:
                    depth_image.image = colored_depth
            if manual.value and time.monotonic()>w.teleop_until:
                vx.value=wz.value=0.
                w.manual_vy=0.
            w.drive.vy=w.manual_vy if manual.value else 0.
            command = w.outputs[-1] if w.outputs else None
            apply_drive_command(w.drive, command, (vx.value,wz.value) if manual.value else None)
            if w.was_paused:
                w.resume_events.append((w.tick,float(w.data.time),w.drive.vx,w.drive.wz))
            w.was_paused = False
            w.runtime.tick(); w.tick+=1
            q=w.data.qpos[w.binding.root_qpos_adr:w.binding.root_qpos_adr+7]
            clearance=course.shaft_clearance(q[0],q[1],course.yaw_of(q[3:]),w.shaft)
            w.min_clearance=min(w.min_clearance,clearance)
            wall_ids={i for i in range(w.model.ngeom) if w.model.geom(i).name.endswith('_wall') or 'rail' in w.model.geom(i).name}
            assert not w.runtime.is_fallen(),'Robot fell; reset required'
            assert clearance>0,'Shaft overlap; reset required'
            assert not any(wall_ids.intersection((c.geom1,c.geom2)) for c in w.data.contact[:w.data.ncon]),'Wall contact; reset required'
            if w.stair.arrived and not manual.value:
                assert w.stair.gen.landings == w.floors*2 and w.drive.vx == w.drive.wz == 0.
                if w.arrival_stop is None:
                    w.arrival_stop = w.runtime.base_pos().copy()
                    sign = 1 if w.direction == 'up' else -1
                    assert w.arrival_stop[0] < -.5 and abs(w.arrival_stop[2] - sign*1.44*w.floors*2 - w.robot.spawn_drop) < .12, ('final pose',w.arrival_stop)
                    w.hold_started = float(w.data.time)
                assert w.stair.gen.landings == w.floors*2 and w.drive.vx == w.drive.wz == 0., 'Nonzero arrival hold'
                if float(w.data.time)-w.hold_started >= 1.5-1e-9:
                    assert np.linalg.norm(w.runtime.base_pos()[:2]-w.arrival_stop[:2]) < .12, 'Stop drift'
                    w.validated = True
                    pause.value=True
        else:
            w.drive.vx=w.drive.vy=w.drive.wz=0.
            w.was_paused = True
        if w.tick%2==0 or pause.value: w.scene.update_from_mjdata(w.data)
        state={'ready':True,'run_id':w.run_id,'cancelled':w.cancelled,'camera_mode':camera_mode.value,'speed':int(sim_speed.value),'time':round(float(w.data.time),2),'paused':pause.value,'manual':manual.value,'lateral_command':w.drive.vy,'floors':w.floors,'landings':w.stair.gen.landings,'status':w.status,'position':w.runtime.base_pos().tolist(),'command':[w.drive.vx,w.drive.wz],'clearance':w.min_clearance if np.isfinite(w.min_clearance) else None,'scene':w.scene_name,'levels':w.levels,'scene_flights':w.scene_flights,'flight_width':w.lane_width,'landing_depth':w.landing_depth,'validation_error':w.error,'direction':w.direction,'validation_passed':w.validated,'resume_events':w.resume_events,'odom_noise':w.noise.metadata(),'odom_error':w.noise_error.tolist(),'estimated_camera_position':None if w.estimated_position is None else w.estimated_position.tolist()}
        presentation_summary.content = (f"### {w.direction.upper()} {w.floors} floors | MuJoCo\n"
            f"Flight / landing: {w.lane_width or 1.8:g} m / {w.landing_depth:g} m\n\n"
            f"Odom: {w.noise.config.mode}, scale {w.noise.config.scale:g}, seed {w.noise.config.seed}")
        if (recording_view.value or camera_mode.value == 'Follow robot') and (w.tick%5==0 or pause.value):
            base = w.runtime.base_pos()
            azimuth = np.radians(105. if base[1]>.7 else 277.)
            offset = np.array([np.cos(azimuth)*1.63,np.sin(azimuth)*1.63,1.63])
            for client in server.get_clients().values():
                client.camera.position = base + offset
                client.camera.look_at = base + (0.,0.,-.16)
                client.camera.up_direction = (0.,0.,1.)
        state_text.content=f"**t={state['time']:.2f}s** | landings **{state['landings']}/{w.floors*2}** | v={w.drive.vx:.2f}, turn={w.drive.wz:.2f}\n\n"+(f'**Validation stopped simulation:** {w.error}. Reset required. This is an independent assertion, not a planner safety stop.' if w.error else w.status)
        if EMBEDDED:
            presentation_summary.content = (f'### {w.direction.upper()} {w.floors} floors | MuJoCo\n'
                f'Flight / landing: {w.lane_width:g} / {w.landing_depth:g} m | Odom: {w.noise.config.mode}, {w.noise.config.scale:g}×, seed {w.noise.config.seed}')
        if recording_view.value or EMBEDDED:
            state_text.content = f"t={state['time']:.1f}s | flights **{w.stair.gen.landings}/{w.floors*2}** | v={w.drive.vx:.2f} | yaw={w.drive.wz:.2f}\n\n" + ('**' + w.error + '**' if w.error else ('**ARRIVED · HOLD PASSED**' if w.validated else ('HOLD CHECK' if w.stair.arrived else w.status.split()[1].upper() if len(w.status.split())>1 else w.status)))
        api.publish(state)
        Path(OUTPUT / 'live-state.json').write_text(json.dumps(state))
        if (w.validated or w.error) and not w.trace_written:
            save_live_trace(w)
    except Exception as exc:
        pause.value=True; w.drive.vx=w.drive.vy=w.drive.wz=0.
        w.error=str(exc)
        state_text.content=f'**Stopped:** {exc}\nUse Start / Reset to rebuild the scene.'
        save_live_trace(w)
        print('LIVE_SIM_STOP',repr(exc),flush=True)
    if api_command:
        api_command.result = api.snapshot()
        api_command.done.set()
    time.sleep(max(.001,(.1 if pause.value else .02/float(sim_speed.value))-(time.perf_counter()-begin)))

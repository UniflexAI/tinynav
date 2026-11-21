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

spec = importlib.util.spec_from_file_location('course', str(ROOT / 'tool/stair_sim_course.py'))
course = importlib.util.module_from_spec(spec)
spec.loader.exec_module(course)
server = viser.ViserServer(host='0.0.0.0', port=8012)
server.gui.configure_theme(control_layout='floating', control_width='medium')
server.gui.add_markdown('## TinyNav stair development\nMuJoCo + Go2 policy + current stair/planner/controller code.\nMouse: orbit / zoom / pan. Start begins a new task in the selected direction.')
scene_choice = server.gui.add_dropdown('Scene', options=('Baseline wide', 'Ten-level narrow'), initial_value='Baseline wide')
scene_note = server.gui.add_markdown('Baseline: original wide course, DOWN 1/2 validated.')
direction_choice = server.gui.add_dropdown('Direction', options=('down', 'up'), initial_value='down')
floors = server.gui.add_dropdown('Floors to traverse', options=('1', '2'), initial_value='2')
landing_depth_choice = server.gui.add_dropdown('Narrow landing depth (m)', options=('1.75', '1.2'), initial_value='1.75')
landing_depth_choice.disabled = True
@scene_choice.on_update
def select_scene(_):
    narrow = scene_choice.value != 'Baseline wide'
    floors.options = ('1', '2', '9') if narrow else ('1', '2')
    landing_depth_choice.disabled = not narrow
    if floors.value not in floors.options:
        floors.value = '2'
    scene_note.content = ('10 full-floor platforms, 18 flights, 1.2 m clear width, Selectable 1.75 / 1.2 m landing depth. Both depths passed UP 9 and DOWN 9 isolated simulation regressions with 18 landings and a zero-command hold. Safety assertions stay enabled. This is not physical-robot validation. Press Start to apply.' if narrow else 'Baseline: original wide course, DOWN 1/2 validated. Press Start to apply.')
start = server.gui.add_button('Start / Reset')
pause = server.gui.add_checkbox('Pause physics', initial_value=True)
manual = server.gui.add_checkbox('Manual drive', initial_value=False)
vx = server.gui.add_slider('Forward m/s', min=-0.3, max=0.3, step=0.05, initial_value=0.0)
wz = server.gui.add_slider('Turn rad/s', min=-0.6, max=0.6, step=0.1, initial_value=0.0)
stop = server.gui.add_button('STOP')
state_text = server.gui.add_markdown('Ready')
bev_planner_image = server.gui.add_image(np.zeros((502,400,3),dtype=np.uint8), label='TinyNav planner BEV (live)')
bev_height_image = server.gui.add_image(np.zeros((502,400,3),dtype=np.uint8), label='Stair height BEV (live)')
depth_image = server.gui.add_image(np.zeros((160,136,3),dtype=np.uint8), label='Front measured depth')
request = {'reset': True, 'run': False}

@start.on_click
def on_start(_):
    request.update(reset=True, run=True)

@stop.on_click
def on_stop(_):
    pause.value = True
    vx.value = wz.value = 0.0

rclpy.init()
world = None

def create_world(n):
    global world
    if world:
        for node in (world.stair, world.planner, world.control):
            node.destroy_node()
        world.camera.close()
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
    stair=stair_node.StairNode(SimpleNamespace(camera_height=.4669,direction=direction,floors=n,landings_per_floor=2,turn='auto',memory=None,rate=2.))
    planner=planning_node.PlanningNode()
    control=cmd_vel_control.CmdVelControlNode()
    clock=SimpleNamespace(monotonic=lambda:float(data.time))
    for module in (stair_node,planning_node,cmd_vel_control): module.time=clock
    stair.K=planner.K=np.array([[303.10,0.,269.37],[0.,303.10,314.67],[0.,0.,1.]])
    control._on_nav_active(Bool(data=True))
    w=SimpleNamespace(model=model,data=data,binding=binding,drive=drive,runtime=runtime,camera=camera,stair=stair,planner=planner,control=control,tick=0,status='Ready',outputs=[],min_clearance=float('inf'),floors=n,bev_result={},bev_planner=None,bev_planner_origin=None,bev_mask=None,bev_path=[],bev_cam_z=0.,bev_frames=0)
    w.levels,w.scene_flights,w.lane_width,w.landing_depth,w.shaft=levels,scene_flights,lane_width,landing_depth,shaft
    w.direction=direction
    w.scene_name=scene_choice.value
    w.error=''
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
    planner.height_map_pub=SimpleNamespace(publish=capture_height)
    obstacle_publisher=planner.obstacle_mask_pub
    def capture_obstacles(msg):
        w.bev_mask=msg
        obstacle_publisher.publish(msg)
    planner.obstacle_mask_pub=SimpleNamespace(publish=capture_obstacles)
    def status(msg):
        w.status=msg.data
        planner._on_stair_status(msg)
        control._on_stair_status(msg)
    stair.status_pub=SimpleNamespace(publish=status)
    stair.target_pub=SimpleNamespace(publish=planner.target_pose_callback)
    stair.poi_change_pub=SimpleNamespace(publish=planner.poi_change_callback)
    def capture_path(msg):
        w.bev_path=[(p.pose.position.x,p.pose.position.y) for p in msg.poses]
        control.path_callback(msg)
    planner.path_pub=SimpleNamespace(publish=capture_path)
    control.cmd_pub=SimpleNamespace(publish=lambda msg:w.outputs.__setitem__(slice(None),[msg]))
    view_model=copy.copy(model)
    view_model.geom_group[view_model.geom_group==3]=2
    w.scene=ViserMujocoScene(server,view_model,num_envs=1)
    w.scene.camera_tracking_enabled=False
    w.scene.update_from_mjdata(data)
    height_bev,planner_bev,_=render_bev(w)
    bev_height_image.image=height_bev
    bev_planner_image.image=planner_bev
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

bridge=CvBridge()
print('LIVE_SIM_READY port=8012',flush=True)
while True:
    begin=time.perf_counter()
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
                pose=np2msg(T,w.stair.get_clock().now().to_msg(),'world','camera')
                pose.header.stamp.sec=int(w.data.time)
                pose.header.stamp.nanosec=int(w.data.time%1*1e9)
                raw=w.camera.capture(w.data)
                depth=bridge.cv2_to_imgmsg(raw,encoding='32FC1'); depth.header=pose.header
                w.stair.odom_callback(pose); w.stair.depth_callback(depth,pose)
                w.control.pose_callback(pose)
                if w.tick%25==0: w.stair.timer_callback()
                with contextlib.redirect_stdout(io.StringIO()): w.planner.sync_callback(depth,pose)
                if w.tick%10==0:
                    height_bev,planner_bev,observed=render_bev(w)
                    bev_height_image.image=height_bev
                    bev_planner_image.image=planner_bev
                    w.bev_frames+=1
                    if w.bev_frames%5==1:
                        cv2.imwrite(str(OUTPUT / 'bev-height.png'),height_bev[...,::-1])
                        cv2.imwrite(str(OUTPUT / 'bev-planner.png'),planner_bev[...,::-1])
                        Path(OUTPUT / 'bev-state.json').write_text(json.dumps({'frames':w.bev_frames,'sim_time':float(w.data.time),'observed_cells':observed,'path_points':len(w.bev_path),'target':None if w.planner.target_pose is None else w.planner.target_pose.tolist()}))
                w.control.cmd_timer_callback()
                cmd=w.outputs[-1] if w.outputs else None
                if not manual.value:
                    w.drive.vx=cmd.linear.x if cmd else 0.
                    w.drive.wz=cmd.angular.z if cmd else 0.
                pixels=(np.clip(raw,0,5)/5*255).astype(np.uint8)
                depth_image.image=cv2.applyColorMap(cv2.resize(pixels,(136,160)),cv2.COLORMAP_TURBO)[...,::-1]
            if manual.value: w.drive.vx,w.drive.wz=vx.value,wz.value
            w.runtime.tick(); w.tick+=1
            q=w.data.qpos[w.binding.root_qpos_adr:w.binding.root_qpos_adr+7]
            clearance=course.shaft_clearance(q[0],q[1],course.yaw_of(q[3:]),w.shaft)
            w.min_clearance=min(w.min_clearance,clearance)
            wall_ids={i for i in range(w.model.ngeom) if w.model.geom(i).name.endswith('_wall') or 'rail' in w.model.geom(i).name}
            assert not w.runtime.is_fallen(),'Robot fell; reset required'
            assert clearance>0,'Shaft overlap; reset required'
            assert not any(wall_ids.intersection((c.geom1,c.geom2)) for c in w.data.contact[:w.data.ncon]),'Wall contact; reset required'
            if w.stair.arrived and not manual.value:
                w.drive.vx=w.drive.wz=0.
                pause.value=True
        else: w.drive.vx=w.drive.wz=0.
        if w.tick%2==0 or pause.value: w.scene.update_from_mjdata(w.data)
        state={'time':round(float(w.data.time),2),'paused':pause.value,'manual':manual.value,'floors':w.floors,'landings':w.stair.gen.landings,'status':w.status,'position':w.runtime.base_pos().tolist(),'command':[w.drive.vx,w.drive.wz],'clearance':w.min_clearance if np.isfinite(w.min_clearance) else None,'scene':w.scene_name,'levels':w.levels,'scene_flights':w.scene_flights,'flight_width':w.lane_width,'landing_depth':w.landing_depth,'validation_error':w.error,'direction':w.direction}
        state_text.content=f"**t={state['time']:.2f}s** | landings **{state['landings']}/{w.floors*2}** | v={w.drive.vx:.2f}, turn={w.drive.wz:.2f}\n\n"+(f'**Validation stopped simulation:** {w.error}. Reset required. This is an independent assertion, not a planner safety stop.' if w.error else w.status)
        Path(OUTPUT / 'live-state.json').write_text(json.dumps(state))
    except Exception as exc:
        pause.value=True; w.drive.vx=w.drive.wz=0.
        w.error=str(exc)
        state_text.content=f'**Stopped:** {exc}\nUse Start / Reset to rebuild the scene.'
        print('LIVE_SIM_STOP',repr(exc),flush=True)
    time.sleep(max(.001,(.1 if pause.value else .02)-(time.perf_counter()-begin)))

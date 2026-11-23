from types import SimpleNamespace
from unittest.mock import Mock
import json
import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParametersAtomically
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.node_manager import BackendNode
from app.backend.routers import nav
from tinynav.core import stair_node, planning_node, vio_safety_node
from tinynav.core.stair_node import StairNode
from tinynav.core.vio_safety_node import VioSafetyNode
from tinynav.platforms.cmd_vel_control import CmdVelControlNode


@pytest.fixture
def nodes(monkeypatch):
    rclpy.init()
    args=SimpleNamespace(camera_height=.45,direction='up',floors=9,landings_per_floor=2,turn='left',memory=None,rate=2.)
    stair=StairNode(args);planner=planning_node.PlanningNode();control=CmdVelControlNode()
    for p in ['path_pub','height_map_pub','obstacle_mask_pub']:setattr(planner,p,Mock())
    stair.status_pub=Mock();stair.poi_change_pub=Mock();control.cmd_pub=Mock()
    control._on_nav_active(Bool(data=True))
    now=[10.]
    clock=SimpleNamespace(monotonic=lambda:now[0])
    for module in (stair_node,planning_node,__import__('tinynav.platforms.cmd_vel_control',fromlist=['time'])):monkeypatch.setattr(module,'time',clock)
    yield stair,planner,control,now
    for node in (stair,planner,control):node.destroy_node()
    rclpy.shutdown()


def test_latch_clears_both_maps_and_preserves_task_count_and_existing_stop(nodes):
    stair,planner,control,now=nodes
    stair.gen.landings=3;stair.gen.landing_ref_z=1.;stair.gen.last_position=np.array([0.,0.,2.])
    stair.gen.landing_waiting_for_turn=True
    stair.gen.frames.append((0.,np.ones((1,3))));planner.occupancy_grid.fill(.2)
    planner.target_pose=np.ones(3);planner.last_T=np.eye(4);planner.reverse_engaged=True
    control.latest_cmd.linear.x=.3
    state=dict(session='one',epoch=1,locked=True,blocked=True,ready=False,reason='meter_jump',fault_id=1,accept_after=10.)
    for node in (stair,planner,control):node.vio_safety_callback(String(data=json.dumps(state)))
    assert not stair.gen.frames and not np.any(planner.occupancy_grid)
    assert planner.target_pose is planner.last_T is None and not planner.reverse_engaged
    assert stair.gen.landings==3 and stair.gen.landing_waiting_for_turn
    stair.pause_callback(Bool(data=False));stair.timer_callback();control.cmd_timer_callback()
    assert stair.vio_blocked and control.cmd_pub.publish.call_args.args[0].linear.x==0.
    # Confirmation enables rebuilding but does not restore the old world-frame target.
    state.update(locked=False,blocked=False)
    for node in (stair,planner,control):node.vio_safety_callback(String(data=json.dumps(state)))
    assert planner.target_pose is None and control.path is None
    from nav_msgs.msg import Odometry
    msg=Odometry();msg.header.stamp.sec=11;msg.pose.pose.position.z=7.
    stair.odom_callback(msg)
    assert stair.gen.landings==3 and stair.gen.landing_ref_z==pytest.approx(6.)


def test_old_paths_and_depth_cannot_repopulate_after_confirmation(nodes):
    stair,planner,control,now=nodes
    state=dict(session='one',epoch=1,locked=False,blocked=False,ready=False,fault_id=1,accept_after=20.)
    for node in (stair,planner,control):node.vio_safety_callback(String(data=json.dumps(state)))
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry,Path
    control.path_callback(Path());planner.sync_callback(Image(),Odometry());planner.target_pose_callback(Odometry())
    stair.depth_callback(Image(),Odometry());stair.odom_callback(Odometry())
    assert not stair.gen.frames and not np.any(planner.occupancy_grid) and control.path is None
    assert planner.target_pose is None


def test_actuator_gate_rejects_nonzero_commands_and_supervisor_loss(monkeypatch):
    from tinynav.platforms import unitree_control as module
    now=[.2];monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:now[0],time=lambda:now[0],sleep=lambda x:None))
    fake=SimpleNamespace(_motion_lock=threading.Lock(),_vio_safety={'blocked':True},_vio_safety_rx=0.,
        sport_client=Mock(),last_twist_time=None,logger=Mock())
    command=Twist();command.linear.x=.3
    module.Ros2UnitreeManagerNode.TwistMessageHandler(fake,command)
    fake.sport_client.Move.assert_not_called()
    module.Ros2UnitreeManagerNode.VioSafetyHandler(fake,String(data='{"blocked":false}'))
    module.Ros2UnitreeManagerNode.TwistMessageHandler(fake,command)
    fake.sport_client.Move.assert_called_once_with(.3,0.,0.)
    now[0]=1.3
    module.Ros2UnitreeManagerNode.TwistMessageHandler(fake,command)
    assert fake.sport_client.Move.call_count==1
    module.Ros2UnitreeManagerNode._vio_watchdog(fake)
    assert fake.sport_client.StopMove.call_count>=3


def test_ros_legacy_confirmation_cannot_bypass_automatic_recovery(monkeypatch):
    rclpy.init()
    guard=VioSafetyNode();api_node=Node('vio_api_test')
    clock=[10.];monkeypatch.setattr(vio_safety_node,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    g=guard.guard;g.status('TRACKING',10.);g.pose(1000.,(0,0,0),10.);g.depth(1000.,10.)
    g.pose(1000.01,(2,0,0),10.01)
    client=api_node.create_client(SetParametersAtomically,'/vio_safety/set_parameters_atomically')
    fake=SimpleNamespace(vio_recovery_status=lambda:g.snapshot(clock[0]),_vio_confirm_client=client)
    fake.cmd_vio_confirm=lambda fid:BackendNode.cmd_vio_confirm(fake,fid)
    monkeypatch.setattr(nav.runner,'node',fake)
    app=FastAPI();app.include_router(nav.router,prefix='/nav')
    executor=MultiThreadedExecutor(num_threads=2);executor.add_node(guard);executor.add_node(api_node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    try:
        assert client.wait_for_service(timeout_sec=2.)
        with TestClient(app) as http:
            assert http.post('/nav/vio/confirm',json={'fault_id':g.fault_id}).status_code==409
            future=client.call_async(SetParametersAtomically.Request(parameters=[Parameter('confirm_fault_id',value=g.fault_id).to_parameter_msg()]))
            deadline=time.monotonic()+2.
            while not future.done() and time.monotonic()<deadline:time.sleep(.01)
            assert future.done() and not future.result().result.successful
            for t in np.arange(10.11,12.62,.1):
                clock[0]=float(t);g.status('TRACKING',float(t));g.pose(1000.+float(t)-10.,(2,0,0),float(t));g.depth(1000.+float(t)-10.,float(t));g.snapshot(float(t))
            assert http.post('/nav/vio/confirm',json={'fault_id':g.fault_id+1}).status_code==409
            assert http.get('/nav/vio').json()['blocked'] is False
            assert http.post('/nav/vio/confirm',json={'fault_id':g.fault_id}).status_code==409
            assert not g.locked
            assert http.post('/nav/vio/confirm',json={'fault_id':True}).status_code==422
    finally:
        executor.shutdown();thread.join(timeout=2.);guard.destroy_node();api_node.destroy_node();rclpy.shutdown()


def test_backend_manual_and_resume_cannot_bypass_latch():
    fake=SimpleNamespace(vio_recovery_status=lambda:dict(locked=True,blocked=True,ready=True),_cmd_vel_pub=Mock())
    fake._require_vio_ready=lambda:BackendNode._require_vio_ready(fake)
    BackendNode.publish_cmd_vel(fake,.5,.2,.5)
    msg=fake._cmd_vel_pub.publish.call_args.args[0]
    assert msg.linear.x==msg.linear.y==msg.angular.z==0.
    with pytest.raises(RuntimeError):BackendNode.cmd_nav_resume(fake)


def test_old_source_frames_cannot_make_recovery_ready():
    rclpy.init()
    guard=VioSafetyNode(source_clock=lambda:100.)
    try:
        from nav_msgs.msg import Odometry
        old=Odometry();old.header.stamp.sec=95
        guard.pose_callback(old)
        assert guard.guard.locked and guard.guard.pose_rx is None
        assert not guard.guard.snapshot(time.monotonic())['ready']
    finally:
        guard.destroy_node();rclpy.shutdown()


def test_late_ros_subscribers_receive_latched_stop_and_clear_maps():
    rclpy.init()
    guard=VioSafetyNode()
    for timer in guard.timers:timer.cancel()
    guard.guard.fault('meter_jump',time.monotonic())
    guard.publish_state()
    args=SimpleNamespace(camera_height=.45,direction='up',floors=9,landings_per_floor=2,turn='auto',memory=None,rate=2.)
    stair=StairNode(args);planner=planning_node.PlanningNode();control=CmdVelControlNode()
    stair.gen.frames.append((0.,np.ones((1,3))));planner.occupancy_grid.fill(.2)
    control.latest_cmd.linear.x=.3;control.cmd_pub=Mock()
    executor=MultiThreadedExecutor(num_threads=2)
    for node in (stair,planner,control):executor.add_node(node)
    try:
        deadline=time.monotonic()+3.
        while time.monotonic()<deadline and any(node._vio_safety is None for node in (stair,planner,control)):
            executor.spin_once(timeout_sec=.05)
        assert all(node._vio_safety and node._vio_safety['locked'] for node in (stair,planner,control))
        assert not stair.gen.frames and not np.any(planner.occupancy_grid)
        assert control.latest_cmd.linear.x==0.
    finally:
        executor.shutdown()
        for node in (guard,stair,planner,control):node.destroy_node()
        rclpy.shutdown()


def test_planner_requires_fresh_occupancy_after_map_reset(nodes):
    stair,planner,control,now=nodes
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    state=dict(session='one',epoch=1,locked=False,blocked=False,ready=False,fault_id=1,accept_after=10.)
    planner.vio_safety_callback(String(data=json.dumps(state)))
    planner.K=np.array([[50.,0.,50.],[0.,50.,50.],[0.,0.,1.]])
    planner.target_pose=np.array([5.,0.,0.])
    pose=Odometry();pose.header.stamp.sec=11;pose.pose.pose.orientation.w=1.
    depth=CvBridge().cv2_to_imgmsg(np.ones((100,100),np.float32),encoding='32FC1');depth.header=pose.header
    planner.path_pub.reset_mock()
    planner.sync_callback(depth,pose)
    planner.path_pub.publish.assert_not_called()
    pose.header.stamp.nanosec=100000000;depth.header=pose.header
    planner.sync_callback(depth,pose)
    planner.path_pub.publish.assert_called_once()

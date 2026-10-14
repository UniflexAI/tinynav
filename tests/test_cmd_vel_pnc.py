import unittest
from types import SimpleNamespace
from unittest.mock import patch
import time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header, Bool
from scipy.spatial.transform import Rotation
from tinynav.core.planning_node import PlanningNode
from tinynav.platforms.cmd_vel_control import CmdVelControlNode

class PncControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()
    def setUp(self):
        self.node = CmdVelControlNode()
        self.sent = []
        self.node.cmd_pub = SimpleNamespace(publish=self.sent.append)
        self.node._nav_active = True
        self.node.pose_callback(Odometry())
    def tearDown(self):
        self.node.destroy_node()
    def path(self, speed=.3, omega=0., stride=1):
        ts=np.arange(31)*.1
        yaw=ts*omega
        xyz=np.zeros((31,3))
        for i in range(1,31):
            xyz[i,:2]=xyz[i-1,:2]+speed*.1*np.array([np.cos(yaw[i]),np.sin(yaw[i])])
        basis=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        quat=Rotation.from_matrix(Rotation.from_euler('z',yaw).as_matrix()@basis).as_quat()
        messages=[]
        header=Header(frame_id='world',stamp=self.node.get_clock().now().to_msg())
        PlanningNode.publish_selected_path(SimpleNamespace(path_pub=SimpleNamespace(publish=messages.append)),np.array([np.column_stack([xyz,quat])]),[0],header,stride=stride)
        return messages[0]
    def test_visual_odometry_subscription_matches_planner(self):
        topics = {subscription.topic_name for subscription in self.node.subscriptions}
        self.assertIn("/slam/odometry_visual", topics)
        self.assertIn("/slam/odometry", topics)

    def test_dense_timing_and_turn(self):
        self.node.path_callback(self.path(.3,.5))
        self.assertTrue(self.node.timed_path)
        self.assertAlmostEqual(self.node.latest_cmd.linear.x,.3,places=2)
        self.assertAlmostEqual(self.node.latest_cmd.angular.z,.5,places=5)
    def test_legacy_sparse_timing(self):
        self.node.path_callback(self.path(.3,0.,10))
        self.assertFalse(self.node.timed_path)
        self.assertAlmostEqual(self.node.latest_cmd.linear.x,.3,places=5)
    def test_reverse_deadzone_matches_plan(self):
        self.node.path_callback(self.path(-.3))
        self.node.cmd_timer_callback()
        self.assertAlmostEqual(self.sent[-1].linear.x,-.3)
        self.node.path_callback(self.path(-.08))
        self.assertEqual(self.sent[-1].linear.x,0.)
    def test_stop_path_stops_without_ramping(self):
        self.node.prev_cmd.linear.x=.3
        self.node.path_callback(self.path(0.))
        self.node.cmd_timer_callback()
        self.assertEqual(self.sent[-1].linear.x,0.)
    def test_path_and_odometry_timeouts_stop(self):
        for kind in ('path','pose'):
            self.node.path_callback(self.path(.3))
            self.node.prev_cmd.linear.x=.3
            if kind=='path':self.node.last_path_update_time=time.monotonic()-2.
            else:self.node.last_pose_update_time=time.monotonic()-2.
            self.node.cmd_timer_callback()
            self.assertEqual(self.sent[-1].linear.x,0.)
            self.node.pose_callback(Odometry())
    def test_repeated_unpaused_status_preserves_acceleration(self):
        self.node.prev_cmd.linear.x=.25
        self.node._on_paused(Bool(data=False))
        self.assertEqual(self.node.prev_cmd.linear.x,.25)

    def test_pause_resume_requires_new_path(self):
        self.node.path_callback(self.path(.3))
        self.node._on_paused(Bool(data=True))
        self.assertEqual(self.sent[-1].linear.x,0.)
        self.node._on_paused(Bool(data=False))
        self.node.cmd_timer_callback()
        self.assertEqual(self.sent[-1].linear.x,0.)
        self.assertIsNone(self.node.last_path_update_time)
    def test_invalid_and_expired_paths_stop(self):
        self.node.path_callback(Path())
        self.assertEqual(self.sent[-1].linear.x,0.)
        msg=self.path(.3)
        for pose in msg.poses:pose.header.stamp.sec-=2
        self.node.path_callback(msg)
        self.assertEqual(self.sent[-1].linear.x,0.)
        msg=self.path(.3);msg.poses[0].pose.orientation.w=float('nan')
        self.node.path_callback(msg)
        self.assertEqual(self.sent[-1].linear.x,0.)
    def test_reverse_sweep_has_true_speed_and_bounded_distance(self):
        from tinynav.core.continuous_planner import ContinuousPlanner
        q=Rotation.from_matrix(np.array([[0,0,1],[-1,0,0],[0,-1,0]])).as_quat()
        out=ContinuousPlanner.reverse(np.zeros(3),q,np.full((100,100),10.),np.array([-2.5,-2.5,0.]),.05,self.node.robot,np.ones((100,100),bool))
        poses,cmd=out
        self.assertAlmostEqual(cmd[0],-.3)
        self.assertAlmostEqual((poses[1,0]-poses[0,0])/.1,-.3)
        self.assertAlmostEqual(poses[-1,0],-.24)
        self.assertAlmostEqual(poses[9,0],poses[8,0])

    def test_near_goal_motion_respects_forward_deadzone(self):
        from tinynav.core.continuous_planner import plan
        q=Rotation.from_matrix(np.array([[0,0,1],[-1,0,0],[0,-1,0]])).as_quat()
        out,_=plan(np.zeros(3),q,np.array([.09,0.,0.]),np.full((100,100),10.),np.array([-2.5,-2.5,0.]),.05,self.node.robot)
        self.assertIsNotNone(out)
        speed=(out[0][1,0]-out[0][0,0])/.1
        self.assertGreaterEqual(speed,self.node.robot.min_linear_vel-.000001)

    def test_map_guidance_follows_route_instead_of_direct_target(self):
        from tinynav.core.navigation_guidance import select_target_position
        route=np.array([[x,0.,0.] for x in np.arange(0.,4.1,.1)]+[[4.,y,0.] for y in np.arange(.1,4.5,.1)])
        selected=select_target_position(route,0,np.zeros(3))
        self.assertEqual(selected[1],0.)
        self.assertGreaterEqual(selected[0],2.5)
        self.assertLess(selected[0],3.)

    def test_unitree_bridge_watchdog_stops_once_on_command_loss(self):
        import ast
        from pathlib import Path
        source=Path(__file__).resolve().parents[1]/'tinynav/platforms/unitree_control.py'
        tree=ast.parse(source.read_text())
        method=next(m for c in tree.body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_check_cmd_timeout')
        namespace={'time':time}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        calls=[]
        bridge=SimpleNamespace(_moving=True,_last_move_mono=time.monotonic(),sport_client=SimpleNamespace(StopMove=lambda:calls.append('stop')))
        check=namespace['_check_cmd_timeout']
        check(bridge);self.assertEqual(calls,[])
        bridge._last_move_mono-=1.
        check(bridge);check(bridge)
        self.assertEqual(calls,['stop'])

    def test_nav_deactivation_sends_stop_then_silence(self):
        self.node._on_nav_active(Bool(data=False))
        count=len(self.sent)
        self.assertEqual(self.sent[-1].linear.x,0.)
        self.node.cmd_timer_callback()
        self.assertEqual(len(self.sent),count)

if __name__=='__main__':unittest.main()

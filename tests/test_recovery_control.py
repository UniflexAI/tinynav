import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import rclpy
from geometry_msgs.msg import Twist
from tinynav.platforms.cmd_vel_control import CmdVelControlNode


class ArbitrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        with patch.dict(os.environ,{'TINYNAV_RULE_RECOVERY':'0'}):
            self.node=CmdVelControlNode()
        self.node.cmd_pub=Mock()
        self.node._nav_active=True
        cmd=Twist();cmd.linear.x=-.2
        self.node.recovery=SimpleNamespace(command=lambda *args:cmd)

    def tearDown(self):
        self.node._nav_active=False
        self.node.destroy_node()

    def test_recovery_is_only_velocity_owner_while_executing(self):
        self.node.cmd_timer_callback()
        self.node.cmd_pub.publish.assert_called_once()
        self.assertEqual(self.node.cmd_pub.publish.call_args.args[0].linear.x,-.2)
        self.assertEqual(self.node.prev_cmd.linear.x,0)

    def test_pause_takes_priority_over_recovery(self):
        self.node._paused=True;self.node.cmd_timer_callback()
        self.assertEqual(self.node.cmd_pub.publish.call_args.args[0].linear.x,0)

    def test_inactive_navigation_remains_silent(self):
        self.node._nav_active=False;self.node.cmd_timer_callback()
        self.node.cmd_pub.publish.assert_not_called()


if __name__=='__main__':unittest.main()

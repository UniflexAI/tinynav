import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from tinynav.core.recovery.ros_adapter import RecoveryAdapter


class AdapterTests(unittest.TestCase):
    def adapter(self,value):
        a=RecoveryAdapter.__new__(RecoveryAdapter)
        a.last_active=True;a.pose_at=None;a.pose_xy=None;a.pose_yaw=None;a.last_status=None
        a.runtime=SimpleNamespace(tick=lambda *args:value,status=lambda:{'phase':'test'})
        a.status_pub=Mock()
        return a

    def test_integer_stop_and_scan_convert_to_ros_float_fields(self):
        for value in [(0,0),(0,.4),(-.2,0)]:
            a=self.adapter(value);cmd=a.command(True,False)
            self.assertEqual(cmd.linear.x,float(value[0]))
            self.assertEqual(cmd.angular.z,float(value[1]))
            a.status_pub.publish.assert_called_once()

    def test_no_recovery_keeps_planner_command_owner(self):
        self.assertIsNone(self.adapter(None).command(True,False))

    def test_small_localization_target_jitter_preserves_history(self):
        from nav_msgs.msg import Odometry
        from tinynav.core.robot_specs import GO2_CONFIG
        from tinynav.core.recovery.runtime import RecoveryRuntime
        a=RecoveryAdapter.__new__(RecoveryAdapter);a.runtime=RecoveryRuntime(GO2_CONFIG)
        a.pose_stamp=None;a.target_anchor=None
        msg=Odometry();msg.header.frame_id='world';msg.pose.pose.position.x=4.0
        a.target(msg);a.runtime.lab.cells[(0,0)]='clear'
        msg.pose.pose.position.x=4.02;a.target(msg)
        self.assertEqual(a.runtime.lab.cells,{(0,0):'clear'})
        msg.pose.pose.position.x=4.5;a.target(msg)
        self.assertEqual(a.runtime.lab.cells,{})



if __name__=='__main__':unittest.main()

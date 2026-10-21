"""Manual ownership and task-retention regression tests. Run in the ROS container."""
import math
import threading
import time
import unittest
from unittest.mock import Mock
from app.backend.node_manager import BackendNode


class ManualInterventionTests(unittest.TestCase):
    def make_node(self, active=False):
        node = BackendNode.__new__(BackendNode)
        node._lock = threading.RLock()
        node.state = 'navigation' if active else 'idle'
        node._nav_active = active
        node._nav_paused = False
        node._manual_intervention = False
        node._manual_velocity = (0.0, 0.0, 0.0)
        node._manual_received = 0.0
        node._cmd_vel_proc, node._planning_proc = object(), object()
        node._map_node_proc = object()
        node._nav_progress = {'poi_index': 2, 'percent': 37}
        node._nav_nodes_running = True
        node._cmd_pois_pub = Mock()
        node._nav_active_pub = Mock()
        node._pub_state = Mock()
        node._pause_pub = Mock()
        node._cmd_vel_pub = Mock()
        node._kill_proc = Mock()
        node._launch_proc = Mock(return_value=object())
        return node

    def test_task_and_map_survive_intervention(self):
        n = self.make_node(active=True)
        map_process = n._map_node_proc
        n.cmd_manual_enter()
        self.assertEqual(n._kill_proc.call_count, 2)
        self.assertIsNone(n._planning_proc)
        self.assertIsNone(n._cmd_vel_proc)
        self.assertIs(n._map_node_proc, map_process)
        n._on_nav_done(type('Done', (), {'data': True})())
        self.assertEqual(n.state, 'navigation')
        self.assertEqual(n._nav_progress['poi_index'], 2)
        with self.assertRaises(ValueError): n.cmd_nav_resume()
        n.cmd_manual_exit()
        self.assertTrue(n._nav_active)
        self.assertTrue(n._nav_paused)
        self.assertEqual(n._launch_proc.call_count, 2)
        n.cmd_nav_resume()
        self.assertFalse(n._nav_paused)

    def test_idle_exit_does_not_create_task(self):
        n = self.make_node()
        n.cmd_manual_enter(); n.cmd_manual_exit()
        self.assertEqual(n.state, 'idle')
        self.assertFalse(n._nav_active)
        self.assertFalse(n._nav_paused)

    def test_watchdog_and_limits(self):
        n = self.make_node()
        n.cmd_manual_enter()
        n.publish_cmd_vel(3, 4, 5)
        n._manual_tick()
        cmd = n._cmd_vel_pub.publish.call_args.args[0]
        self.assertAlmostEqual(math.hypot(cmd.linear.x, cmd.linear.y), .5)
        self.assertEqual(cmd.angular.z, 1)
        n._manual_received = time.monotonic() - .31
        n._manual_tick()
        cmd = n._cmd_vel_pub.publish.call_args.args[0]
        self.assertEqual((cmd.linear.x, cmd.linear.y, cmd.angular.z), (0, 0, 0))
        n.publish_cmd_vel(float('nan'), 0, 0)
        self.assertTrue(all(math.isfinite(x) for x in n._manual_velocity))

    def test_explicit_cancel_during_manual_is_final(self):
        n = self.make_node(active=True)
        n.cmd_manual_enter()
        n.cmd_nav_cancel()
        n.cmd_manual_exit()
        self.assertEqual(n.state, 'idle')
        self.assertFalse(n._nav_active)
        self.assertFalse(n._nav_paused)

    def test_legacy_command_cannot_bypass_manual_ownership(self):
        n = self.make_node(active=True)
        n.cmd_manual_enter()
        n._cmd_cb(type('Command', (), {'data': 'navigation'})())
        self.assertEqual(n.state, 'navigation')
        self.assertTrue(n._manual_intervention)
        self.assertEqual(n._launch_proc.call_count, 0)
        with self.assertRaises(ValueError): n.cmd_start_nav_nodes()

    def test_teleop_cannot_override_active_task(self):
        n = self.make_node(active=True)
        n.publish_cmd_vel(.5, 0, 0)
        n._cmd_vel_pub.publish.assert_not_called()


if __name__ == '__main__': unittest.main()

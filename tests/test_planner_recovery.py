import unittest
import numpy as np
from tinynav.core.planning_node import StallRecovery, generate_trajectory_library_3d, score_trajectories_by_ESDF, generate_recovery_trajectory
from scipy.spatial.transform import Rotation as R


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.r = StallRecovery()
        self.xy = np.array([0., 0.])
        self.target = np.array([4., 0.])

    def tick(self, t, yaw=0., xy=None, safe=lambda *args: True):
        return self.r.command(t, self.xy if xy is None else np.array(xy), yaw, self.target, safe)

    def scan(self):
        for t in np.arange(0., 6.2, .1): self.tick(t)

    def test_stall_then_full_measured_rotation(self):
        self.scan()
        self.assertEqual(self.r.phase, 'scan')
        self.assertEqual(self.tick(6.3), (0., -.4))
        for k in range(1, 160):
            self.tick(6.3+k*.1, (k*.04+np.pi)%(2*np.pi)-np.pi)
        self.assertEqual(self.r.phase, 'retreat')
        self.assertEqual(self.tick(22.5, .04), (-.2, 0.))

    def test_moving_does_not_trigger(self):
        for t in np.arange(0., 10., .1): self.tick(t, xy=[t*.2,0.])
        self.assertEqual(self.r.phase, 'normal')

    def test_blocked_rotation_does_not_trigger(self):
        for t in np.arange(0., 8., .1): self.tick(t, safe=lambda *args:False)
        self.assertEqual(self.r.phase, 'normal')
        self.assertEqual(self.r.attempts, 0)

    def test_new_obstacle_aborts_and_target_resets(self):
        self.scan()
        self.assertEqual(self.tick(6.3,safe=lambda *args:False), (0.,0.))
        self.assertEqual(self.r.phase, 'normal')
        self.target = np.array([5.,0.])
        self.tick(7.)
        self.assertEqual(self.r.attempts, 0)

    def test_arrival_stops(self):
        self.assertEqual(self.tick(0., xy=[3.8,0.]), (0.,0.))
        self.assertEqual(self.r.phase, 'arrived')

    def test_scan_timeout_and_repeated_attempt_budget(self):
        self.scan()
        self.assertEqual(self.tick(32.), (0.,0.))
        self.assertEqual(self.r.phase, 'normal')
        self.r.attempts=3
        for t in np.arange(40., 48., .1):self.tick(t)
        self.assertEqual(self.r.phase, 'normal')

    def test_retreat_turn_probe_and_resume(self):
        self.r.phase='retreat';self.r.target=self.target;self.r.anchor=self.xy;self.r.phase_at=0.
        self.assertEqual(self.tick(1.,xy=[-3.01,0.]),(0.,-.4))
        self.assertEqual(self.r.phase,'turn')
        self.assertEqual(self.tick(2.,yaw=np.pi/2,xy=[-3.01,0.]),(.2,0.))
        self.assertEqual(self.r.phase,'probe')
        self.assertIsNone(self.tick(3.,yaw=np.pi/2,xy=[-3.01,1.81]))
        self.assertEqual(self.r.phase,'normal')

    def test_sensor_gap_aborts_and_missing_goal_resets(self):
        self.scan()
        self.assertEqual(self.tick(8.),(0.,0.))
        self.assertEqual(self.r.phase,'normal')
        self.r.command(8.1,self.xy,0.,None,lambda *args:True)
        self.assertEqual(self.r.attempts,0)

    def test_scan_selects_clear_retreat_heading(self):
        self.r.phase='scan';self.r.rotation=2*np.pi;self.r.target=self.target;self.r.phase_at=0.
        def safe(v,w,heading=0.,duration=3.):
            return not (v < 0 and abs(heading) < .1)
        self.tick(1.,safe=safe)
        self.assertEqual(self.r.phase,'align_retreat')
        self.assertAlmostEqual(abs(self.r.anchor),np.pi/4)

    def test_recovery_heading_and_velocity_match_existing_controller(self):
        q=R.from_matrix([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]]).as_quat()
        traj,param=generate_recovery_trajectory(np.zeros(3),q,.2,0.,np.pi/2)
        self.assertAlmostEqual(traj[10,1]-traj[0,1],.2)
        self.assertAlmostEqual(traj[10,0]-traj[0,0],0.)
        traj,param=generate_recovery_trajectory(np.zeros(3),q,0.,-.4)
        fwd0=R.from_quat(traj[0,3:]).as_matrix()[:,2]
        fwd1=R.from_quat(traj[10,3:]).as_matrix()[:,2]
        delta=np.arctan2(fwd1[1],fwd1[0])-np.arctan2(fwd0[1],fwd0[0])
        self.assertAlmostEqual(delta,.4)

    def test_recovery_path_keeps_original_stride_and_collision_checker(self):
        ts,ps=generate_trajectory_library_3d(num_samples=3,max_linear_vel=0.,max_angular_vel=.4)
        k=np.argmin(np.linalg.norm(ps-np.array([0.,-.4]),axis=1))
        poses=ts[k][::10]
        self.assertEqual(len(poses),4)
        self.assertTrue(np.allclose(poses[:,:3],poses[0,:3]))
        scores,_=score_trajectories_by_ESDF(ts[k:k+1],np.zeros((100,100),dtype=np.float32),np.array([-2.5,-2.5,-.25]),.05)
        self.assertTrue(np.isinf(scores[0]))


if __name__=='__main__':unittest.main()

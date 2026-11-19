import unittest

import numpy as np

from tinynav.platforms.path_tracking import PID, PathTracker, wrap_angle


class PathTrackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PathTracker(1.0, 0.75)

    def test_straight_feedforward(self):
        self.tracker.set_path([[0, 0, 0], [0.5, 0, 0]], [0, 0], 1.0)
        np.testing.assert_allclose(self.tracker.command([0.1, 0, 0], 0.2, 0.1), [0.5, 0])

    def test_recovers_from_displacement_and_heading_error(self):
        for side in [-1, 1]:
            with self.subTest(side=side):
                tracker = PathTracker(1.0, 0.75)
                pose = np.array([0.0, side * 0.5, side * 0.3])
                dt = 0.05
                for tick in range(240):
                    x = tick * dt * 0.5
                    tracker.set_path([[x, 0, 0], [x + 0.5, 0, 0]], [0, 0], 1.0)
                    v, w = tracker.command(pose, 0, dt)
                    self.assertTrue(0 <= v <= 1.0 and abs(w) <= 0.75)
                    pose += dt * np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), w])
                self.assertLess(abs(pose[1]), 0.05)
                self.assertLess(abs(wrap_angle(pose[2])), 0.05)
                self.assertLess(abs(pose[0] - 6.0), 0.05)

    def test_tracks_curved_trajectory(self):
        pose = np.array([0.0, -0.3, -0.2])
        dt = 0.05

        def reference(t):
            return [2 * np.sin(t / 4), 2 * (1 - np.cos(t / 4)), t / 4]

        for tick in range(240):
            t = tick * dt
            self.tracker.set_path([reference(t + i * 0.2) for i in range(6)],
                                  [i * 0.2 for i in range(6)], 1.0)
            v, w = self.tracker.command(pose, 0, dt)
            pose += dt * np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), w])
        self.assertLess(np.linalg.norm(pose[:2] - reference(12)[:2]), 0.08)
        self.assertLess(abs(wrap_angle(pose[2] - 3)), 0.08)

    def test_world_frame_rotation_and_translation_invariance(self):
        poses = np.array([[0, 0, 0], [0.5, 0, 0]])
        current = np.array([0.1, 0.2, 0.15])
        self.tracker.set_path(poses, [0, 0], 1.0)
        expected = self.tracker.command(current, 0.2, 0.1)
        angle = 1.4
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        poses[:, :2] = poses[:, :2] @ rotation.T + [14, -7]
        poses[:, 2] += angle
        current[:2] = rotation @ current[:2] + [14, -7]
        current[2] += angle
        self.tracker.reset()
        self.tracker.set_path(poses, [0, 0], 1.0)
        np.testing.assert_allclose(self.tracker.command(current, 0.2, 0.1), expected)

    def test_timestamp_spacing_takes_precedence(self):
        self.tracker.set_path([[0, 0, 0], [0.1, 0, 0], [0.3, 0, 0]], [10, 10.2, 10.6], 1)
        target, v, w = self.tracker.sample(0.4)
        np.testing.assert_allclose(target, [0.2, 0, 0])
        self.assertAlmostEqual(v, 0.5)
        self.assertEqual(w, 0)

    def test_repeated_or_backwards_stamps_use_planner_spacing(self):
        for stamps in [[0, 0], [1, 0]]:
            with self.subTest(stamps=stamps):
                self.tracker.set_path([[0, 0, 0], [0.5, 0, 0]], stamps, 1)
                np.testing.assert_allclose(self.tracker.sample(0.5)[0], [0.25, 0, 0])

    def test_yaw_interpolates_across_wrap(self):
        self.tracker.set_path([[0, 0, np.deg2rad(179)], [0, 0, np.deg2rad(-179)]], [0, 1], 1)
        target, _, yaw_rate = self.tracker.sample(0.5)
        self.assertAlmostEqual(abs(target[2]), np.pi)
        self.assertAlmostEqual(yaw_rate, np.deg2rad(2))
        _, command = self.tracker.command([0, 0, np.deg2rad(-179.9)], 0.5, 0.1)
        self.assertLess(abs(command), 0.1)

    def test_stops_at_last_pose_without_extrapolation(self):
        self.tracker.set_path([[0, 0, 0], [0.5, 0, 0]], [0, 1], 1)
        np.testing.assert_allclose(self.tracker.command([0.5, 0, 0], 10, 0.1), [0, 0])

    def test_turns_before_driving_when_facing_backwards(self):
        self.tracker.set_path([[0, 0, 0], [0.5, 0, 0]], [0, 1], 1)
        v, w = self.tracker.command([0, 0, 2.0], 0, 0.1)
        self.assertEqual(v, 0)
        self.assertLess(w, 0)

    def test_integral_is_bounded_and_resettable(self):
        pid = PID([0, 1, 0])
        for _ in range(100):
            self.assertLessEqual(abs(pid.update(100, 0.1)), 0.5)
        pid.reset()
        self.assertEqual(pid.update(0, 0.1), 0)

    def test_invalid_configuration_and_path_are_rejected(self):
        for gains in [[1, 2], [1, -1, 0], [1, np.nan, 0]]:
            with self.subTest(gains=gains), self.assertRaises(ValueError):
                PID(gains)
        for poses in [[[0, 0, 0]], [[0, 0, 0], [np.nan, 0, 0]]]:
            with self.subTest(poses=poses), self.assertRaises(ValueError):
                self.tracker.set_path(poses, [0, 1], 1)


if __name__ == '__main__':
    unittest.main()

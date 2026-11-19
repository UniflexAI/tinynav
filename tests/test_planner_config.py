import unittest

from tinynav.core.planner_config import planner_mode, policy_engine_path, route_guidance


class PlannerSelectionTests(unittest.TestCase):
    def test_public_names_and_legacy_names(self):
        for name, expected in (("library", "library"), ("dwa", "library"),
                               ("rl", "rl"), ("ppo", "rl"), (" PPO ", "rl")):
            with self.subTest(name=name):
                self.assertEqual(planner_mode({"TINYNAV_PLANNER": name}), expected)
        self.assertEqual(planner_mode({}), "library")
        with self.assertRaises(ValueError):
            planner_mode({"TINYNAV_PLANNER": "unknown"})

    def test_ppo_selects_route_guidance(self):
        self.assertTrue(route_guidance({"TINYNAV_PLANNER": "ppo"}))
        self.assertTrue(route_guidance({"TINYNAV_PLANNER": "ppo", "TINYNAV_RL_ROUTE_GUIDANCE": "0"}))
        self.assertFalse(route_guidance({"TINYNAV_PLANNER": "rl"}))
        self.assertTrue(route_guidance({"TINYNAV_PLANNER": "rl", "TINYNAV_RL_ROUTE_GUIDANCE": "1"}))
        self.assertFalse(route_guidance({"TINYNAV_PLANNER": "dwa", "TINYNAV_RL_ROUTE_GUIDANCE": "1"}))

    def test_engine_matches_selected_model(self):
        self.assertTrue(policy_engine_path({"TINYNAV_PLANNER": "ppo"}).endswith(
            "/tinynav/models/rl_nav_policy_ppo_20261114.plan"))
        self.assertIn("/rl_nav_policy_v7_", policy_engine_path({"TINYNAV_PLANNER": "rl"}))
        for name in ("ppo", "rl"):
            self.assertEqual(policy_engine_path({"TINYNAV_PLANNER": name,
                                                 "TINYNAV_RL_ENGINE": "/models/test.plan"}), "/models/test.plan")


if __name__ == "__main__":
    unittest.main()

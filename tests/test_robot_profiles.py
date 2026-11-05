"""RobotProfile table: audit-backed facts per model, and the policy maths.

The profile table is the contract the emu conformance check mirrors; these
tests pin the values the driver and charge watch rely on. AS2 facts come from
the 2026-10-08 real-dog audit (docs/multi-robot-sim-testbench.md §2).
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault('ROBOT_TYPE', 'b2')

from tinynav.platforms import unitree_control as uc  # noqa: E402
from tinynav.platforms import robot_profiles as rp  # noqa: E402


class TestProfileTable(unittest.TestCase):
    def test_type_values_are_the_robot_type_strings(self):
        self.assertEqual(rp.RobotType.GO2.value, 'go2')
        self.assertEqual(rp.RobotType.AS2.value, 'as2')
        self.assertEqual(rp.RobotType.G1.value, 'g1')

    def test_every_type_has_a_profile(self):
        for t in rp.RobotType:
            self.assertIn(t, rp.PROFILES)

    def test_as2_is_audit_shaped(self):
        p = rp.PROFILES[rp.RobotType.AS2]
        self.assertEqual(p.family, rp.Family.QUADRUPED)
        self.assertEqual(p.motion_api, 'sport')
        self.assertEqual(p.lowstate_idl, 'unitree_hg')
        self.assertEqual(p.battery.topic, 'rt/lf/bmsstate')
        self.assertEqual(p.battery.idl, 'unitree_hg')
        self.assertFalse(p.battery.has_status)
        self.assertIsInstance(p.battery.policy, rp.CurrentSignPolicy)
        self.assertEqual(p.odom, 'sportmodestate')
        # NO per-motion gait assertion: BalanceStand is a toggle (audit), and
        # the driver re-arms at every motion start -- an unconditional assert
        # would walk NORMAL_WALK back to STAND_UP and the next Move gets -1.
        self.assertIsNone(p.gait_assertion)
        self.assertFalse(p.needs_gait_worker)
        self.assertEqual(p.stand_steps, ('StandUp', 'BalanceStand'))
        self.assertEqual(p.sit_steps, ('StandDown',))

    def test_as2w_mirrors_as2_on_the_wheeled_family(self):
        a2 = rp.PROFILES[rp.RobotType.AS2]
        w = rp.PROFILES[rp.RobotType.AS2W]
        self.assertEqual(w.family, rp.Family.QUADRUPED_WHEELED)
        for field in ('motion_api', 'lowstate_idl', 'odom',
                      'gait_assertion', 'needs_gait_worker',
                      'stand_steps', 'sit_steps'):
            self.assertEqual(getattr(w, field), getattr(a2, field), field)
        self.assertEqual(w.battery.topic, a2.battery.topic)
        self.assertEqual(w.battery.idl, a2.battery.idl)
        self.assertEqual(w.battery.has_status, a2.battery.has_status)
        self.assertIsInstance(w.battery.policy, rp.CurrentSignPolicy)

    def test_go2_family_uses_the_go_surface(self):
        # GO2W's gait assertion is deliberately None: the wheeled base has no
        # walk gait for the machinery to assert, so no gait worker either.
        for t, gait, worker in ((rp.RobotType.GO2, 'classic_walk', True),
                                (rp.RobotType.GO2W, None, False),
                                (rp.RobotType.B2, 'classic_walk', True),
                                (rp.RobotType.B2W, 'classic_walk', True)):
            p = rp.PROFILES[t]
            self.assertEqual(p.lowstate_idl, 'unitree_go', t)
            self.assertEqual(p.battery.topic, 'rt/lf/lowstate', t)
            self.assertTrue(p.battery.has_status, t)
            self.assertIsInstance(p.battery.policy, rp.StatusBitPolicy, t)
            self.assertEqual(p.gait_assertion, gait, t)
            self.assertEqual(p.needs_gait_worker, worker, t)
        # The real b2 never publishes rt/utlidar/robot_odom (2026-10-15 rig
        # probe), so its chassis odometry is republished from
        # rt/lf/sportmodestate like the AS2's. b2w is still unprobed.
        self.assertEqual(rp.PROFILES[rp.RobotType.B2].odom, 'sportmodestate')
        self.assertEqual(rp.PROFILES[rp.RobotType.GO2].odom, 'robot_odom')
        self.assertEqual(rp.PROFILES[rp.RobotType.GO2W].odom, 'robot_odom')
        self.assertEqual(rp.PROFILES[rp.RobotType.B2W].odom, 'robot_odom')
        # SwitchGait must NOT be a stand row: the driver appends it to every
        # stand, and a SwitchGait(1) there is what a b2 stand got wrong
        # (2444767). The machinery appends the gait in force instead.
        for t in (rp.RobotType.GO2, rp.RobotType.GO2W, rp.RobotType.B2,
                  rp.RobotType.B2W):
            self.assertNotIn('SwitchGait', rp.PROFILES[t].stand_steps, t)

    def test_g1_carries_no_battery_and_no_odom(self):
        p = rp.PROFILES[rp.RobotType.G1]
        self.assertEqual(p.family, rp.Family.HUMANOID)
        self.assertEqual(p.motion_api, 'loco')
        self.assertIsNone(p.battery)
        self.assertIsNone(p.odom)
        self.assertEqual(p.stand_steps, ('Damp', 'Squat2StandUp'))
        self.assertEqual(p.sit_steps, ('StandUp2Squat',))

    def test_as2_threshold_is_not_the_go_family_one(self):
        """AS2 rests at -970 mA; the go threshold (-1000) would sit inside
        that and misjudge a resting dog as pad-powered."""
        as2_thr = rp.AS2_EXTERNAL_POWER_THRESHOLD_MA
        go_thr = rp.GO_EXTERNAL_POWER_THRESHOLD_MA
        self.assertGreater(as2_thr, -970)
        self.assertGreaterEqual(as2_thr, go_thr + 800)

    def test_lookup_by_string_is_case_insensitive(self):
        self.assertIs(rp.get_profile(' AS2 '), rp.PROFILES[rp.RobotType.AS2])
        self.assertIsNone(rp.get_profile('lekiwi'))
        self.assertIsNone(rp.get_profile(''))
        with self.assertRaises(ValueError):
            rp.require_profile('go3')


class _FakeClient:
    def __init__(self):
        self.calls = []

    def _rec(name):
        def _m(self, *args):
            self.calls.append((name, args))
            return 0
        return _m

    StandUp = _rec('StandUp')
    StandDown = _rec('StandDown')
    BalanceStand = _rec('BalanceStand')
    ClassicWalk = _rec('ClassicWalk')
    SwitchGait = _rec('SwitchGait')
    Damp = _rec('Damp')
    Squat2StandUp = _rec('Squat2StandUp')
    Move = _rec('Move')


class TestStepResolution(unittest.TestCase):
    """Profile step names resolve against the client the profile's factory
    would build -- without touching DDS."""

    def _node(self, profile):
        node = uc.Ros2UnitreeManagerNode.__new__(uc.Ros2UnitreeManagerNode)
        node.profile = profile
        node.sport_client = _FakeClient()
        return node

    def test_as2_stand_then_sit(self):
        node = self._node(rp.PROFILES[rp.RobotType.AS2])
        for name in node.profile.stand_steps:
            node._resolve_step(name)()
        self.assertEqual(node.sport_client.calls,
                         [('StandUp', ()), ('BalanceStand', ())])

    def test_b2_stand_tail_carries_switchgait_with_its_argument(self):
        node = self._node(rp.PROFILES[rp.RobotType.B2])
        # a stand carries the gait in force only when a gait worker exists
        # (`self.gait is not None` gates the append) -- set what __init__ and a
        # first gait would have set
        node.gait = object()
        node.walk_gait = 'classic'
        node._gait_steps = {'classic': ('SwitchGait',
                                        lambda: node.sport_client.SwitchGait(1))}
        for name in node.profile.stand_steps:
            node._resolve_step(name)()
        tail = node._gait_steps[node.walk_gait]
        tail[1]()
        self.assertEqual(node.sport_client.calls[-1], ('SwitchGait', (1,)))

    def test_go2_classicwalk_asserts_true(self):
        node = self._node(rp.PROFILES[rp.RobotType.GO2])
        node._resolve_step('ClassicWalk')()
        self.assertEqual(node.sport_client.calls,
                         [('ClassicWalk', (True,))])

    def test_g1_squat_waits_before_standing(self):
        node = self._node(rp.PROFILES[rp.RobotType.G1])
        call = node._resolve_step('Squat2StandUp')
        self.assertIs(call.__func__,
                      uc.Ros2UnitreeManagerNode._squat_to_stand)


class TestPolicyMaths(unittest.TestCase):
    def test_status_bit_policy_maps_the_enum(self):
        pol = rp.StatusBitPolicy(-1000.0)
        self.assertEqual(pol.state_from(rp.CHG, None), rp.CHARGING)
        self.assertEqual(pol.state_from(rp.DCHG, None), rp.DISCHARGING)
        self.assertEqual(pol.state_from(rp.SAFE, None), rp.IDLE)
        self.assertEqual(pol.state_from(99, None), rp.UNKNOWN)

    def test_status_bit_charging_short_circuits_external_power(self):
        pol = rp.StatusBitPolicy(-1000.0)
        self.assertIs(pol.externally_powered(rp.CHARGING, True, False, None), True)
        self.assertIs(pol.externally_powered(rp.CHARGING, False, False, None), None)
        self.assertIs(pol.externally_powered(rp.DCHG, True, False, -35), None)
        self.assertIs(pol.externally_powered(rp.DCHG, True, True, -35), True)
        self.assertIs(pol.externally_powered(rp.DCHG, True, True, -1468), False)

    def test_current_sign_policy_judges_by_threshold(self):
        pol = rp.CurrentSignPolicy(-100.0)
        self.assertEqual(pol.state_from(None, None), rp.UNKNOWN)
        self.assertEqual(pol.state_from(None, 1500), rp.CHARGING)
        self.assertEqual(pol.state_from(None, -970), rp.DISCHARGING)
        self.assertIs(pol.externally_powered(rp.CHARGING, True, True, 1500), True)
        self.assertIs(pol.externally_powered(rp.DISCHARGING, True, True, -970),
                      False)
        self.assertIs(pol.externally_powered(rp.DISCHARGING, True, False, -970),
                      None)
        self.assertIs(pol.externally_powered(rp.DISCHARGING, False, True, -970),
                      None)


if __name__ == '__main__':
    unittest.main()

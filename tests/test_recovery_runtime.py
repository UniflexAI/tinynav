import unittest
from tinynav.core.robot_specs import GO2_CONFIG
from tinynav.core.recovery.runtime import RecoveryRuntime


class RuntimeTests(unittest.TestCase):
    def ready(self):
        r=RecoveryRuntime(GO2_CONFIG);r.reset([4,0,0])
        r.lab.cells={(x,y):'clear' for x in range(-50,51) for y in range(-50,51)}
        r.depth_at=10;r.last_tick=9.9
        r.lab.history.extend((10-i/10,[0,0],0,4) for i in range(80,0,-1))
        return r

    def test_no_sensor_cannot_start_recovery(self):
        r=self.ready();r.depth_at=None
        self.assertIsNone(r.tick([0,0],0,10,True,False,0))
        self.assertIsNone(r.policy.scan)

    def test_stall_starts_scan_and_keeps_override(self):
        r=self.ready();a=r.tick([0,0],0,10,True,False,0)
        self.assertEqual(a,(0,.4))
        r.depth_at=10.1;b=r.tick([0,0],2,10.1,True,False,0)
        self.assertEqual(b,(0,.4))
        self.assertEqual(r.policy.scans,1)

    def test_final_scan_turn_remains_physically_executable(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        r.policy.scan['rotation_deg']=359.0;r.depth_at=10.1
        self.assertEqual(r.tick([0,0],0,10.1,True,False,0),(0,.1))

    def test_pause_stops_owned_command(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        self.assertEqual(r.tick([0,0],0,10.1,True,True,0),(0,0))
        self.assertIsNone(r.policy.scan)
        self.assertEqual(len(r.lab.history),0)

    def test_stale_sensor_stops_scan(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        self.assertEqual(r.tick([0,0],0,10.6,True,False,0),(0,0))
        self.assertEqual(r.phase,'stale_sensor')

    def test_unknown_rotation_footprint_rejects_scan(self):
        r=self.ready();r.lab.cells.clear()
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        self.assertIsNone(r.policy.scan)

    def test_timer_gap_stops_scan(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0);r.depth_at=10.3
        self.assertEqual(r.tick([0,0],0,10.3,True,False,0),(0,0))
        self.assertEqual(r.phase,'timer_gap')

    def test_target_change_discards_previous_recovery(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0);r.reset([8,0,0])
        self.assertIsNone(r.policy.scan);self.assertEqual(r.policy.scans,0)
        self.assertEqual(r.lab.cells,{})

    def test_arrival_owns_stop_without_later_restart(self):
        r=self.ready();r.target=[0,0,0]
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        r.depth_at=10.1
        self.assertEqual(r.tick([.5,0],0,10.1,True,False,0),(0,0))
        self.assertEqual(r.policy.scans,0)
        self.assertEqual(r.tick([.5,0],0,11,True,False,1),(0,0))

    def test_scan_then_fresh_observation_executes_without_model(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0);r.policy.scan['rotation_deg']=360
        r.depth_at=10.1;self.assertEqual(r.tick([0,0],0,10.1,True,False,0),(0,0))
        r.depth_at=10.2
        cmd=r.tick([0,0],0,10.2,True,False,0)
        self.assertLess(cmd[0],0)
        self.assertIsNotNone(r.executor.active)
        self.assertFalse(r.status()['model_used'])

    def test_long_detour_requires_every_swept_cell_observed(self):
        from tinynav.core.recovery.recovery_strategies import proposals, poses
        from tinynav.core.recovery.rule_recovery import recovery_candidates
        r=self.ready()
        offered=proposals([0,0],0,[4,0,0],r.robot,r.lab.cells,.1,[])
        plan=next(p for p in offered if p['id']=='retreat_300_left')
        endpoint=list(poses([0,0],0,plan['stages']))[-1][0]
        self.assertAlmostEqual(endpoint[0],-3)
        self.assertAlmostEqual(endpoint[1],1.8)
        result=recovery_candidates([plan],[0,0],0,r.robot,r.lab.cells,.1)
        self.assertEqual(result['plans'],[plan])
        r.lab.cells.pop((-30,18))
        self.assertEqual(recovery_candidates([plan],[0,0],0,r.robot,r.lab.cells,.1)['plans'],[])

    def test_longer_clear_side_preferred_when_attempt_counts_equal(self):
        from tinynav.core.recovery.observed_retreat_selection import choose
        short={'id':'retreat_180_right','stages':[{'linear_mps':-.2,'duration_s':9}]}
        long={'id':'retreat_300_left','stages':[{'linear_mps':-.2,'duration_s':15}]}
        self.assertEqual(choose({'plans':[short,long]})['id'],long['id'])
        self.assertEqual(choose({'plans':[short,long]},[{'strategy_id':long['id']}])['id'],short['id'])

    def test_near_goal_finish_uses_observed_footprint_and_original_goal(self):
        r=self.ready();r.target=[.44,0,0]
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(.1,0))
        self.assertEqual(r.phase,'goal_approach')
        self.assertEqual(r.target,[.44,0,0])
        self.assertEqual(r.tick([0,0],0,10.1,True,True,0),(0,0))

    def test_near_goal_finish_never_drives_into_unknown(self):
        r=self.ready();r.target=[.44,0,0];r.lab.cells.clear()
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        self.assertFalse(r.approach_active)

    def test_near_goal_finish_is_bounded_to_one_attempt(self):
        r=self.ready();r.target=[.44,0,0];r.tick([0,0],0,10,True,False,0)
        r.last_tick=25.9;r.depth_at=26
        self.assertEqual(r.tick([0,0],0,26,True,False,0),(0,0))
        self.assertEqual(r.phase,'goal_approach_timeout')
        self.assertTrue(r.approach_attempted);self.assertFalse(r.approach_active)

    def test_new_obstacle_stops_entire_remaining_stage(self):
        r=self.ready()
        p={'id':'retreat','duration_s':3,'stages':[{'name':'retreat','linear_mps':-.2,'yaw_radps':0,'duration_s':3}]}
        r.executor.start(p,[0,0],0,[4,0,0],9.9)
        r.lab.cells[(-5,0)]='blocked'
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        self.assertEqual(r.executor.memory[-1]['outcome'],'updated_footprint_rejected')


if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace

import numpy as np

from tinynav.core.planner_recovery import (
    MotionTracker, ObservationMap, RecoveryExecutor, RecoveryRuntime,
    advance, choose, proposals, recovery_candidates, shortlist,
)
from tinynav.core.robot_specs import GO2_CONFIG


def ready_runtime():
    r = RecoveryRuntime(GO2_CONFIG); r.reset([4, 0, 0])
    r.map.cells = {(x, y): 'clear' for x in range(-50, 51) for y in range(-50, 51)}
    r.depth_at = 10; r.last_tick = 9.9
    r.tracker.history.extend((10-i/10, [0., 0.]) for i in range(80, 0, -1))
    return r


class RuleTest(unittest.TestCase):
    def test_unique_clear_progress_pivot_prevents_unnecessary_retreat(self):
        robot={'length':.4,'width':.3}
        cells={(i,j):'clear' for i in range(-30,31) for j in range(-20,21)}
        def plan(identity,v):
            return {'id':identity,'predicted_goal_progress_m':.2,'stages':[{'linear_mps':v,'yaw_radps':0,'duration_s':2}]}
        right=plan('pivot_right',.3);retreat=plan('retreat_60_left',-.3)
        selected=recovery_candidates([right,retreat],[0,0],0,robot,cells,.1)
        self.assertEqual(choose(selected)['id'],right['id'])
        left=plan('pivot_left',.3)
        selected=recovery_candidates([right,left,retreat],[0,0],0,robot,cells,.1)
        self.assertEqual(choose(selected)['id'],left['id'])

    def test_observed_pivot_fallback_requires_measured_sweep(self):
        robot={'length':.4,'width':.3}
        cells={(i,j):'clear' for i in range(-30,31) for j in range(-20,21)}
        plans=[{'id':'pivot_'+side,'predicted_goal_progress_m':progress,'stages':[{'linear_mps':.3,'yaw_radps':0,'duration_s':2}]} for side,progress in [('left',.2),('right',.3)]]
        result=recovery_candidates(plans,[0,0],0,robot,cells,.1)
        self.assertEqual(choose(result)['id'],'pivot_right')
        cells.clear()
        self.assertEqual(recovery_candidates(plans,[0,0],0,robot,cells,.1)['plans'],[])


class MotionTrackerTests(unittest.TestCase):
    def test_stall_requires_stationary_anchor(self):
        t=MotionTracker()
        t.update([0,0],10)
        self.assertFalse(t.stalled(11.9,2.0))
        self.assertTrue(t.stalled(12.0,2.0))
        t.update([.2,0],13)  # moved past the epsilon: anchor resets
        self.assertFalse(t.stalled(14.5,2.0))
        self.assertTrue(t.stalled(15.0,2.0))

    def test_recent_reports_windowed_path_length(self):
        t=MotionTracker()
        for i in range(20):
            t.update([i*.01,0],10+i*.2)   # 4 s of creeping, 0.19 m path
        window,moved=t.recent(14)
        self.assertGreaterEqual(window,3.8);self.assertAlmostEqual(moved,.19,places=1)
        t.update([0,0],30)                # gap: old history falls out of the window
        window,moved=t.recent(30)
        self.assertEqual(window,0);self.assertEqual(moved,0)

    def test_reset_clears_history_and_anchor(self):
        t=MotionTracker();t.update([0,0],10)
        t.reset()
        self.assertEqual(len(t.history),0);self.assertIsNone(t.anchor);self.assertIsNone(t.anchor_since)


class ObservationMapTests(unittest.TestCase):
    def config(self):
        return {'camera': {'fx': 1, 'fy': 1}, 'robot': {'obstacle': {'robot_z_bottom': -0.4, 'robot_z_top': 0.4}}}

    def test_measured_ray_carves_free_and_blocks_endpoint(self):
        m=ObservationMap(); cfg=self.config()
        self.assertFalse(m.cells)
        T=np.array([[0,0,1,0],[1,0,0,0],[0,-1,0,0.4],[0,0,0,1]],float)
        m.observe(np.array([[2.]],float),T,cfg['camera'],cfg['robot'],now=100.0)
        self.assertEqual(m.cells[(20,0)],'blocked')
        self.assertEqual(m.cells[(10,0)],'clear')
        self.assertNotIn((25,0),m.cells)
        m.observe(np.zeros((1,1)),T,cfg['camera'],cfg['robot'],now=100.1)  # blank frame adds nothing
        self.assertNotIn((25,0),m.cells)

    def test_clear_expires_but_blocked_is_retained(self):
        m=ObservationMap(); cfg=self.config()
        m.cells={(0,0):'clear',(1,0):'blocked'}; m.observed_at={(0,0):10.0,(1,0):10.0}
        m.observe(np.zeros((1,1)),np.eye(4),cfg['camera'],cfg['robot'],now=41.0)
        self.assertNotIn((0,0),m.cells)
        self.assertEqual(m.cells[(1,0)],'blocked')


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.robot = {'shape':'circle','radius':.2,'length':.4,'width':.3}
        self.cells = {(i,j):'clear' for i in range(-30,31) for j in range(-15,16)}
        self.plans = [{'id':f'retreat_{n}_{side}','stages':[{'linear_mps':-.3,'yaw_radps':0,'duration_s':n/30}]}
                      for side in ('left','right') for n in (60,120,180)]

    def test_keep_short_clear_retreats_for_cost_ranking(self):
        result = shortlist(self.plans,[0,0],0,self.robot,self.cells,.1)
        self.assertEqual(len(result['plans']),6)
        self.assertEqual(choose(result,[{'strategy_id':'retreat_60_left'}])['id'],'retreat_60_right')

    def test_single_unknown_cell_excludes_long_sweep(self):
        del self.cells[(-17,0)]
        result = shortlist(self.plans,[0,0],0,self.robot,self.cells,.1)
        self.assertEqual(len(result['plans']),4)
        self.assertNotIn('retreat_180_left',[p['id'] for p in result['plans']])

    def test_blocked_current_footprint_refuses_every_retreat(self):
        self.cells[(0,0)] = 'blocked'
        result = shortlist(self.plans,[0,0],0,self.robot,self.cells,.1)
        self.assertEqual(result['plans'],[])
        self.assertIsNone(choose(result))


class RecoveryTests(unittest.TestCase):
    robot = {'max_linear_vel':1,'max_angular_vel':.75,'length':.4,'width':.3,'shape':'square'}
    def test_observed_blockage_and_memory(self):
        plans = proposals([0,0],0,[3,3],self.robot,{},.1,[])
        self.assertEqual(len(plans),8)
        self.assertTrue(all(p['unknown_fraction']==1 for p in plans))
        memory = [{'strategy_id':plans[0]['id'],'start_xy':[0,0],'start_yaw_deg':0}]
        self.assertNotIn(plans[0]['id'],[p['id'] for p in proposals([0,0],0,[3,3],self.robot,{},.1,memory)])
        blocked = {(i,j):'blocked' for i in range(-25,26) for j in range(-25,26)}
        self.assertEqual(proposals([0,0],0,[3,3],self.robot,blocked,.1,[]),[])
    def test_stages_abort_before_motion_and_memory(self):
        plan={'id':'test','duration_s':2,'stages':[{'name':'retreat','linear_mps':-1,'yaw_radps':0,'duration_s':1}, {'name':'probe','linear_mps':1,'yaw_radps':0,'duration_s':1}]}
        e=RecoveryExecutor();e.start(plan,[0,0],0,[2,0],0)
        self.assertEqual(e.step([0,0],0,[2,0],.1,.1,.1,lambda p,y:p[0]<-.4),(0,0,.1))
        self.assertIsNone(e.active);self.assertEqual(e.memory[0]['outcome'],'stage_collision_rejected')
        self.assertEqual(plan['stages'][0]['duration_s'],1)
        e=RecoveryExecutor();e.start(plan,[0,0],0,[2,0],0);xy=[0,0];yaw=0
        for i in range(20):
            v,w,dt=e.step(xy,yaw,[2,0],.1,(i+1)*.1,.1,lambda p,y:False)
            xy,yaw=advance(xy,yaw,v,w,dt)
        self.assertIsNone(e.active);self.assertEqual(e.memory[0]['outcome'],'completed')
        self.assertAlmostEqual(xy[0],0)
        self.assertGreater(e.settle_until,2)
    def test_mid_stage_new_obstacle_and_stale_report(self):
        p={'id':'test','duration_s':1,'stages':[{'name':'probe','linear_mps':.3,'yaw_radps':0,'duration_s':1}]}
        e=RecoveryExecutor();e.start(p,[0,0],0,[2,0],0)
        self.assertIsNotNone(e.step([0,0],0,[2,0],.1,.1,.1,lambda p,y:False))
        self.assertEqual(e.step([.03,0],0,[2,0],.1,.2,.1,lambda p,y:p[0]>.04),(0,0,.1))
        self.assertEqual(e.memory[-1]['outcome'],'collision_risk')
        e=RecoveryExecutor();e.start(p,[0,0],0,[2,0],0);e.step([0,0],0,[2,0],.1,.1,1,lambda p,y:False)
        self.assertEqual(e.memory[-1]['outcome'],'stale_plan')
    def test_rotating_footprint_sweep_not_only_center_ray(self):
        from tool.simulator.planning_scene import SimObject, robot_hits_objects
        objects=[SimObject('corner','box',[.05,.23,.5],[.03,.03,1])]
        collision=lambda xy,yaw:robot_hits_objects(xy,yaw,self.robot,objects)
        self.assertFalse(collision([0,0],0))
        plan={'id':'turn','duration_s':3,'stages':[{'name':'turn','linear_mps':0,'yaw_radps':.6,'duration_s':3}]}
        e=RecoveryExecutor();e.start(plan,[0,0],0,[2,0],0)
        self.assertEqual(e.step([0,0],0,[2,0],.1,.1,.1,collision),(0,0,.1))
        self.assertEqual(e.memory[-1]['outcome'],'stage_collision_rejected')


class RuntimeTests(unittest.TestCase):
    def ready(self):
        return ready_runtime()

    def test_no_sensor_cannot_start_recovery(self):
        r=self.ready();r.depth_at=None
        self.assertIsNone(r.tick([0,0],0,10,True,False,0))
        self.assertIsNone(r.scan)

    def test_stall_starts_scan_and_keeps_override(self):
        r=self.ready()
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,.4))
        r.depth_at=10.1
        self.assertEqual(r.tick([0,0],2,10.1,True,False,0),(0,.4))
        self.assertEqual(r.scans,1)

    def test_young_history_does_not_start_scan(self):
        r=self.ready();r.tracker.history.clear()
        self.assertIsNone(r.tick([0,0],0,10,True,False,0))
        self.assertIsNone(r.scan)
        self.assertEqual(r.phase,'watching')

    def test_final_scan_turn_remains_physically_executable(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        r.scan['rotation_deg']=359.0;r.depth_at=10.1
        self.assertEqual(r.tick([0,0],0,10.1,True,False,0),(0,.1))

    def test_pause_stops_owned_command(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        self.assertEqual(r.tick([0,0],0,10.1,True,True,0),(0,0))
        self.assertIsNone(r.scan)
        self.assertEqual(len(r.tracker.history),0)

    def test_stale_sensor_stops_scan(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        self.assertEqual(r.tick([0,0],0,10.6,True,False,0),(0,0))
        self.assertEqual(r.phase,'stale_sensor')

    def test_unknown_rotation_footprint_rejects_scan(self):
        r=self.ready();r.map.cells.clear()
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        self.assertEqual(r.phase,'scan_footprint_unknown_or_blocked')

    def test_timer_gap_stops_recovery(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0);r.depth_at=10.3
        self.assertEqual(r.tick([0,0],0,10.3,True,False,0),(0,0))
        self.assertEqual(r.phase,'timer_gap')

    def test_cancel_releases_scan_and_does_not_execute(self):
        r=self.ready()
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,.4))
        self.assertEqual(r.phase,'scanning')
        self.assertEqual(r.tick([0,0],0,10.1,False,False,0),(0,0))
        self.assertIsNone(r.scan);self.assertIsNone(r.executor.active)

    def test_drift_aborts_active_scan(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        self.assertEqual(r.tick([.03,0],0,10.1,True,False,0),(0,0))
        self.assertEqual(r.phase,'scan_aborted');self.assertIsNone(r.scan)

    def test_completed_scan_waits_for_new_report(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0)
        r.scan['rotation_deg']=359.6
        self.assertEqual(r.tick([0,0],0,10.1,True,False,0),(0,0))
        self.assertEqual(r.phase,'fresh_report')
        self.assertEqual(r.tick([0,0],0,10.2,True,False,0),(0,0))
        self.assertIsNone(r.last_selection);self.assertIsNone(r.executor.active)
        r.depth_at=10.25
        r.tick([0,0],0,10.3,True,False,0)
        self.assertIsNotNone(r.last_selection);self.assertIsNotNone(r.executor.active)

    def test_scan_budget_limits_attempts(self):
        r=self.ready();r.scans=3
        self.assertIsNone(r.tick([0,0],0,10,True,False,0))
        self.assertEqual(r.phase,'attempt_limit');self.assertIsNone(r.scan)

    def test_stale_report_cannot_start_scan(self):
        r=self.ready();r.depth_at=10
        self.assertIsNone(r.tick([0,0],0,11,True,False,0))
        self.assertIsNone(r.scan)

    def test_target_change_discards_previous_recovery(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0);r.reset([8,0,0])
        self.assertIsNone(r.scan);self.assertEqual(r.scans,0)
        self.assertEqual(r.map.cells,{})

    def test_arrival_owns_stop_without_later_restart(self):
        r=self.ready();r.target=[0,0,0]
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        r.depth_at=10.1
        self.assertEqual(r.tick([.5,0],0,10.1,True,False,0),(0,0))
        self.assertEqual(r.scans,0)
        self.assertEqual(r.tick([.5,0],0,11,True,False,1),(0,0))

    def test_scan_then_fresh_observation_executes(self):
        r=self.ready();r.tick([0,0],0,10,True,False,0);r.scan['rotation_deg']=360
        r.depth_at=10.1;self.assertEqual(r.tick([0,0],0,10.1,True,False,0),(0,0))
        r.depth_at=10.2
        cmd=r.tick([0,0],0,10.2,True,False,0)
        self.assertNotEqual(cmd,(0,0))
        self.assertIsNotNone(r.executor.active)
        self.assertEqual(r.status()['decision_source'],'rule')

    def test_long_detour_requires_every_swept_cell_observed(self):
        from tinynav.core.planner_recovery import poses
        r=self.ready()
        offered=proposals([0,0],0,[4,0,0],r.robot,r.map.cells,.1,[])
        plan=next(p for p in offered if p['id']=='retreat_300_left')
        endpoint=list(poses([0,0],0,plan['stages']))[-1][0]
        self.assertAlmostEqual(endpoint[0],-3)
        self.assertAlmostEqual(endpoint[1],1.8)
        result=recovery_candidates([plan],[0,0],0,r.robot,r.map.cells,.1)
        self.assertEqual(result['plans'],[plan])
        r.map.cells.pop((-30,18))
        self.assertEqual(recovery_candidates([plan],[0,0],0,r.robot,r.map.cells,.1)['plans'],[])

    def test_shorter_clear_side_preferred_when_attempt_counts_equal(self):
        short={'id':'retreat_180_right','stages':[{'linear_mps':-.2,'duration_s':9}]}
        long={'id':'retreat_300_left','stages':[{'linear_mps':-.2,'duration_s':15}]}
        self.assertEqual(choose({'plans':[short,long]})['id'],short['id'])

    def test_near_goal_finish_uses_observed_footprint_and_original_goal(self):
        r=self.ready();r.target=[.44,0,0]
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(.1,0))
        self.assertEqual(r.phase,'goal_approach')
        self.assertEqual(r.target,[.44,0,0])
        self.assertEqual(r.tick([0,0],0,10.1,True,True,0),(0,0))

    def test_near_goal_finish_never_drives_into_unknown(self):
        r=self.ready();r.target=[.44,0,0];r.map.cells.clear()
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
        r.map.cells[(-5,0)]='blocked'
        self.assertEqual(r.tick([0,0],0,10,True,False,0),(0,0))
        self.assertEqual(r.executor.memory[-1]['outcome'],'updated_footprint_rejected')


class RankedRecoveryTests(unittest.TestCase):
    def plan(self,n,progress=0):
        return {'id':'retreat_'+str(n),'predicted_goal_progress_m':progress,
                'stages':[{'linear_mps':-.2,'yaw_radps':0,'duration_s':n/.2}]}

    def test_short_then_long_after_reblocked(self):
        plans=[self.plan(n) for n in (.6,1.2,1.8,3.)]
        self.assertEqual(choose({'plans':plans})['id'],'retreat_0.6')
        memory=[{'strategy_id':'retreat_0.6','outcome':'reblocked','retreat_m':.6}]
        self.assertEqual(choose({'plans':plans},memory)['id'],'retreat_1.2')
        memory.append({'strategy_id':'retreat_1.2','outcome':'reblocked','retreat_m':1.2})
        self.assertEqual(choose({'plans':plans},memory)['id'],'retreat_3.0')

    def test_progress_can_outweigh_small_extra_distance(self):
        a=self.plan(.6,-.5);b=self.plan(1.2,.6)
        self.assertEqual(choose({'plans':[a,b]})['id'],b['id'])

    def test_reblocked_is_detected_after_settling(self):
        r=ready_runtime()
        r.executor.memory=[{'strategy_id':'retreat_60_left',
            'outcome':'completed','start_xy':[0,0],'end_xy':[-.6,0],'retreat_m':.6}]
        r.tick([0,0],0,10,True,False,0)
        self.assertEqual(r.executor.memory[-1]['outcome'],'reblocked')

    def test_failed_short_cannot_make_unknown_long_eligible(self):
        robot={'length':.4,'width':.3};cells={(x,y):'clear' for x in range(-10,11) for y in range(-10,11)}
        result=recovery_candidates([self.plan(.6),self.plan(3.)],[0,0],0,robot,cells,.1)
        self.assertNotIn('retreat_3.0',[p['id'] for p in result['plans']])


class HandoffCostTests(unittest.TestCase):
    def test_dead_end_prefers_exit_over_short_interior_motion(self):
        from tinynav.core.planner_recovery import estimate_handoff
        r=RecoveryRuntime(GO2_CONFIG)
        cells={(x,y):'clear' for x in range(-65,66) for y in range(-50,51)}
        for x in range(-10,21):
            cells[(x,14)]='blocked';cells[(x,-14)]='blocked'
        for y in range(-14,15):cells[(20,y)]='blocked'
        plans=proposals([.9,0],0,[4,0,0],r.robot,cells,.1,[])
        result=recovery_candidates(plans,[.9,0],0,r.robot,cells,.1)
        estimate_handoff(result['plans'],[.9,0],0,[4,0,0],r.robot,cells,.1)
        self.assertTrue(choose(result)['id'].startswith('retreat_300_'))

    def test_open_space_prefers_short_progress_pivot(self):
        from tinynav.core.planner_recovery import estimate_handoff
        r=RecoveryRuntime(GO2_CONFIG)
        cells={(x,y):'clear' for x in range(-65,66) for y in range(-50,51)}
        plans=proposals([0,0],0,[0,4,0],r.robot,cells,.1,[])
        result=recovery_candidates(plans,[0,0],0,r.robot,cells,.1)
        estimate_handoff(result['plans'],[0,0],0,[0,4,0],r.robot,cells,.1)
        self.assertEqual(choose(result)['id'],'pivot_left')


class PlannerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rclpy
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        import rclpy
        rclpy.shutdown()

    def test_observed_scan_command_uses_existing_path_convention(self):
        import time
        from scipy.spatial.transform import Rotation as R
        from tinynav.core.planning_node import PlanningNode, generate_recovery_trajectory
        node=PlanningNode()
        try:
            node.K=np.array([[100.,0.,80.],[0.,100.,50.],[0.,0.,1.]])
            now=time.monotonic();r=node.recovery;r.reset([4.,0.,0.])
            r.map.cells={(x,y):'clear' for x in range(-50,51) for y in range(-50,51)}
            r.map.observed_at={c:now for c in r.map.cells}
            r.tracker.history.extend((now-i/10,[0.,0.]) for i in range(80,0,-1))
            T=np.eye(4);T[:3,:3]=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
            # Set camera offset so the control center is exactly the history center.
            from tinynav.core.robot_specs import ROBOT_CONFIG
            T[:3,3]=T[:3,:3]@ROBOT_CONFIG.cam_offset_3d
            stamp=node.get_clock().now().nanoseconds*1e-9
            value=node.recovery_command(np.zeros((100,160),dtype=np.float32),T,stamp)
            self.assertEqual(value,(0.,.4))
            traj,_=generate_recovery_trajectory(node.camera_to_robot_center(T),R.from_matrix(T[:3,:3]).as_quat(),value[0],-value[1])
            q0=R.from_quat(traj[0,3:]);q1=R.from_quat(traj[10,3:])
            body=np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
            delta=R.from_matrix((q0.as_matrix()@body).T@(q1.as_matrix()@body)).as_rotvec()[2]
            self.assertAlmostEqual(delta,.4)
            node.paused_callback(type('Flag',(),{'data':True})())
            self.assertIsNone(r.scan)
            self.assertIsNone(node.recovery_command(np.zeros((100,160),dtype=np.float32),T,stamp))
        finally:node.destroy_node()


if __name__ == '__main__':
    unittest.main()

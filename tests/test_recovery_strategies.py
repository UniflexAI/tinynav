import copy
import unittest
from tool.simulator.recovery_strategies import RecoveryExecutor, proposals, advance

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

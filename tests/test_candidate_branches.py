import copy
import unittest
from tool.simulator.candidate_branches import compact_report, rollout
from tool.simulator.decision_observer import planning_questions


def candidate(i,v,w,cost=1,reasons=None):
    return {'id':i,'linear_mps':v,'angular_radps':w,'cost':cost,'endpoint_world_xy':[0,0],
            'obstacle_score':0 if cost is not None else None,'reasons':reasons or []}


class BranchTests(unittest.TestCase):
    def test_diversity_and_camera_yaw_sign(self):
        report={'selected_id':0,'notes':[],'candidates':[candidate(0,0,0),candidate(1,.2,0),candidate(2,.2,-.4),candidate(3,.2,.4),candidate(4,-.2,0,1e9,['reverse_gate_penalty']),candidate(5,.5,0,None,['sampled_footprint_collision'])]}
        compact=compact_report(report)
        self.assertEqual(compact['representatives']['left']['candidate_id'],2)
        self.assertEqual(compact['representatives']['forward']['collision_rejected'],1)
        self.assertEqual(len(compact['top_candidates']),5)
        questions=planning_questions({'planning':compact})
        self.assertIn('candidate_4',questions['alternative']['criteria'])
        self.assertNotIn('candidate_5',questions['alternative']['criteria'])

    def test_rollouts_isolated_and_collision_detected(self):
        config={'target':[4,0,0],'objects':[], 'robot':{'length':.2,'width':.2,'max_linear_vel':1,'max_angular_vel':1,'control_x':0,'control_y':0}}
        seed={'xy':[0,0],'yaw_deg':0};original=copy.deepcopy((config,seed))
        forward=rollout(config,seed,candidate(1,.5,0))
        self.assertAlmostEqual(forward['goal_progress_m'],1.5)
        commanded=candidate(8,.5,0);commanded['control_linear_mps']=.4;commanded['control_yaw_radps']=0
        self.assertAlmostEqual(rollout(config,seed,commanded)['goal_progress_m'],1.2)
        left=rollout(config,seed,candidate(2,.5,-.3))
        self.assertGreater(left['final_xy'][1],0)
        self.assertEqual((config,seed),original)
        config['objects']=[{'name':'wall','kind':'box','center':[.7,0,.5],'size':[.1,2,1]}]
        self.assertTrue(rollout(config,seed,candidate(1,.5,0))['collision'])
        with self.assertRaises(ValueError):rollout(config,seed,candidate(9,.5,0,None))
        config['map_path']='unsupported'
        with self.assertRaises(ValueError):rollout(config,seed,candidate(1,.5,0))

if __name__=='__main__':unittest.main()

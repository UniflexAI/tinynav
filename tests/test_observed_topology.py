import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
from tool.simulator.observed_topology import components,topology,capture_audit
from tool.simulator.decision_information import enrich
from tool.simulator.decision_observer import planning_questions
from tool.simulator.recovery_strategies import proposals
from tool.simulator.navigation_lab import NavigationLab

class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.robot={'length':.4,'width':.3,'max_linear_vel':1,'max_angular_vel':.75}
        self.state={'planning':{'top_candidates':[],'selected_id':0},'recovery':{'strategies':proposals([0,0],0,[3,0],self.robot,{},.1,[]),'previous_attempts':[]}}
        self.samples=[{'xy':[-.9,0],'yaw_deg':0,'t':0},{'xy':[0,0],'yaw_deg':0,'t':4}]
    def test_unknown_gap_not_connected(self):
        labels,sizes=components({(0,0):'clear',(2,0):'clear',(1,0):'blocked'})
        self.assertNotEqual(labels[(0,0)],labels[(2,0)]);self.assertEqual(sorted(sizes.values()),[1,1])
        labels,_=components({(0,0):'clear',(2,0):'clear'})
        self.assertNotEqual(labels[(0,0)],labels[(2,0)])
    def test_same_choices_commands_and_unknown_not_free(self):
        rich=enrich(self.state,[0,0],0,[3,0],self.robot,{},.1,self.samples);before=copy.deepcopy(rich)
        v2=topology(rich,[0,0],0,[3,0],self.robot,{},.1,self.samples,{},20)
        self.assertEqual(rich,before);self.assertEqual(planning_questions(v2),planning_questions(rich))
        self.assertEqual(v2['spatial_context']['route']['goal_center_connection'],'unknown')
        self.assertEqual(v2['spatial_context']['route']['past_route_footprint']['unknown'],1)
        self.assertEqual(v2['spatial_context']['route']['recent_clear_fraction'],0)
        for a,b in zip(rich['recovery']['strategies'],v2['recovery']['strategies']):self.assertEqual(a['stages'],b['stages'])
    def test_freshness_without_reclassifying_stale_clear(self):
        cells={(i,j):'clear' for i in range(-30,41) for j in range(-30,31)}
        rich=enrich(self.state,[0,0],0,[3,0],self.robot,cells,.1,self.samples)
        old=topology(rich,[0,0],0,[3,0],self.robot,cells,.1,self.samples,{c:0 for c in cells},20)
        fresh=topology(rich,[0,0],0,[3,0],self.robot,cells,.1,self.samples,{c:19 for c in cells},20)
        self.assertEqual(old['spatial_context']['route']['recent_clear_fraction'],0)
        self.assertEqual(fresh['spatial_context']['route']['recent_clear_fraction'],1)
        self.assertEqual(old['spatial_context']['route']['goal_center_connection'],'observed_connected')
        plan=next(p for p in fresh['recovery']['strategies'] if p['id']=='retreat_60_left')
        self.assertGreater(float(plan['observation_evidence']['endpoint'].split(',')[0]),0)
        self.assertEqual(plan['observation_evidence']['endpoint'].split(',')[1],'connected')
        self.assertEqual(cells[(0,0)],'clear')
    def test_cell_timestamps_do_not_refresh_old_blocked_with_free_ray(self):
        lab=NavigationLab();camera={'fx':2,'fy':2};depth=np.ones((4,4));transform=np.eye(4)
        with patch('tool.simulator.navigation_lab.time.monotonic',return_value=1):lab.observe_depth(depth,transform,camera,self.robot)
        self.assertTrue(lab.cells);cell=next(c for c,v in lab.cells.items() if v=='clear')
        lab.cells[cell]='blocked';lab.cell_observed_at[cell]=1
        with patch('tool.simulator.navigation_lab.time.monotonic',return_value=20):lab.observe_depth(depth,transform,camera,self.robot)
        self.assertEqual(lab.cells[cell],'blocked');self.assertEqual(lab.cell_observed_at[cell],1)
        self.assertTrue(any(t==20 for c,t in lab.cell_observed_at.items() if lab.cells[c]=='clear'))
        lab.reset();self.assertEqual(lab.cell_observed_at,{})

    def test_numpy_cell_keys_serialize_in_audit(self):
        cells={(np.int64(1),np.int64(2)):'clear'}
        audit=capture_audit(cells,{(1,2):np.float64(10)},[0,0],.1,20)
        self.assertEqual(json.loads(json.dumps(audit))['cells'],[[1,2,'clear',10.0]])

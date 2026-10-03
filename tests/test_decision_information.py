import copy
import unittest
from tool.simulator.decision_information import enrich,basic
from tool.simulator.decision_observer import planning_questions
from tool.simulator.recovery_strategies import proposals

class InformationTests(unittest.TestCase):
    def setUp(self):
        self.robot={'length':.4,'width':.3,'max_linear_vel':1,'max_angular_vel':.75}
        self.state={'planning':{'top_candidates':[],'selected_id':0},'recovery':{'strategies':proposals([0,0],0,[3,0],self.robot,{},.1,[]),'previous_attempts':[]}}
        self.samples=[{'xy':[-1,0],'yaw_deg':0,'t':0},{'xy':[-.5,0],'yaw_deg':0,'t':2},{'xy':[0,0],'yaw_deg':0,'t':4}]
    def test_information_only_and_unknown_not_reclassified(self):
        source=copy.deepcopy(self.state)
        rich=enrich(self.state,[0,0],0,[3,0],self.robot,{},.1,self.samples)
        self.assertEqual(self.state,source);self.assertEqual(basic(rich),source)
        self.assertEqual(planning_questions(rich),planning_questions(source))
        self.assertEqual(rich['spatial_context']['trajectory_breadcrumbs'][0].split(',')[:2],['-1.0','0.0'])
        retreat=next(p for p in rich['recovery']['strategies'] if p['id']=='retreat_60_left')['observation_evidence']['stages'][0]
        self.assertEqual(float(retreat.split(',')[2]),1)
        self.assertGreater(float(retreat.split(',')[3]),.9)
        self.assertEqual(len(rich['spatial_context']['local_grid']['rows']),9)
    def test_grid_rotation_and_coordinate_frame(self):
        cells={(0,16):'blocked'}
        rich=enrich(self.state,[0,0],90,[0,3],self.robot,cells,.1,[])
        self.assertEqual(rich['spatial_context']['goal_xy_m'],[3,0])
        self.assertEqual(rich['spatial_context']['local_grid']['rows'][0][4],'X')
    def test_bounded_history_and_stage_evidence(self):
        samples=[{'xy':[i*.3,0],'yaw_deg':0,'t':i} for i in range(1000)]
        rich=enrich(self.state,[0,0],0,[3,0],self.robot,{},.1,samples)
        self.assertLessEqual(len(rich['spatial_context']['trajectory_breadcrumbs']),4)
        for old,new in zip(self.state['recovery']['strategies'],rich['recovery']['strategies']):
            self.assertEqual(old['stages'],new['stages']);self.assertEqual(old['id'],new['id'])
            self.assertEqual(len(new['observation_evidence']['stages']),len(old['stages']))

    def test_unused_single_action_rows_compacted_and_baseline_restored(self):
        source=copy.deepcopy(self.state)
        source['planning'].update(top_candidates=[{'id':0,'cost':1,'reasons':[]},{'id':1,'cost':2,'reasons':[]}],collision_examples=[{'id':2}])
        rich=enrich(source,[0,0],0,[3,0],self.robot,{},.1,self.samples)
        self.assertEqual([c['id'] for c in rich['planning']['top_candidates']],[0])
        self.assertNotIn('collision_examples',rich['planning'])
        self.assertEqual(basic(rich,source['planning']),source)
        self.assertEqual(planning_questions(rich),planning_questions(source))

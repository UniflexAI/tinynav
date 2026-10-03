import copy,time,unittest
from tool.simulator.experiment_policy import guarded_command

class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.c={'id':2,'linear_mps':.2,'angular_radps':-.3,'control_linear_mps':.19,'control_yaw_radps':.3,'cost':4,'reasons':[]}
        self.report={'robot_world_xy':[0,0],'robot_yaw_deg':0,'candidates':[self.c]}
        self.record={'status':'complete','created_at_unix':100,'context':{'config_generation':3,'planning_report':copy.deepcopy(self.report)},'request':{'questions':{'alternative':{'criteria':{'candidate_2':'test'}}}},'response':{'answers':{'alternative':{'choice':'candidate_2'}}}}
    def check(self,record=None,report=None,xy=None,yaw=0,generation=3,now=101,age=.1):
        return guarded_command(record or self.record,report or self.report,xy or [0,0],yaw,generation,now,age)
    def test_command_duration_and_no_mutation(self):
        original=copy.deepcopy(self.record);command,reason=self.check()
        self.assertEqual(reason,'applied');self.assertEqual(command['duration_s'],.75);self.assertEqual(command['linear_mps'],.19);self.assertEqual(self.record,original)
    def test_fallbacks(self):
        self.assertEqual(self.check(generation=4)[1],'scene_changed')
        self.assertEqual(self.check(now=109)[1],'expired_result')
        self.assertEqual(self.check(age=1)[1],'stale_plan')
        self.assertEqual(self.check(xy=[1,0])[1],'pose_changed')
        self.assertEqual(self.check(yaw=20)[1],'pose_changed')
        changed=copy.deepcopy(self.report);changed['candidates'][0]['cost']=None
        self.assertEqual(self.check(report=changed)[1],'collision_rejected')
        changed=copy.deepcopy(self.report);changed['candidates'][0]['linear_mps']=.5
        self.assertEqual(self.check(report=changed)[1],'candidate_changed')
        r=copy.deepcopy(self.record);r['response']['answers']['alternative']['choice']='uncertain'
        self.assertEqual(self.check(record=r)[1],'uncertain')
        r=copy.deepcopy(self.record);r['request']['questions']['alternative']['criteria']={}
        self.assertEqual(self.check(record=r)[1],'not_offered')

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from tool.simulator.navigation_lab import NavigationLab

class LabTests(unittest.TestCase):
    def config(self):
        return {'target':[4,0,0], 'camera':{'fx':1,'fy':1}, 'robot':{'obstacle':{'robot_z_bottom':-0.4,'robot_z_top':0.4}}, 'objects':[]}

    def test_unknown_and_observed_occlusion(self):
        lab = NavigationLab(); cfg=self.config()
        self.assertEqual(lab.world_state([0,0],0,cfg)['directions']['behind']['ends_in'],'unknown')
        T=np.array([[0,0,1,0],[1,0,0,0],[0,-1,0,0.4],[0,0,0,1]],float)
        lab.observe_depth(np.array([[2.]],float),T,cfg['camera'],cfg['robot'])
        ray=lab.world_state([0,0],0,cfg)['directions']['ahead']
        self.assertEqual(ray['ends_in'],'blocked')
        self.assertLess(ray['known_clear_distance_m'],2)
        self.assertNotIn((25,0),lab.cells)
        blank=NavigationLab(); blank.observe_depth(np.zeros((1,1)),T,cfg['camera'],cfg['robot'])
        self.assertFalse(blank.cells)

    def test_body_band_and_relative_bearing(self):
        cfg=self.config();cfg['objects']=[{'center':[1,0,2],'size':[.2,.2,.2]}]
        lab=NavigationLab()
        state=lab.world_state([0,0],90,cfg,'full_scene')
        self.assertEqual(state['goal']['bearing'],'right')
        self.assertEqual(state['directions']['right']['ends_in'],'range_limit')
        cfg['objects'][0]['center'][2]=.4
        self.assertEqual(lab.world_state([0,0],0,cfg,'full_scene')['directions']['ahead']['ends_in'],'blocked')

    def test_metrics_outcomes_and_export(self):
        cfg=self.config();lab=NavigationLab();lab.begin(cfg,[0,0],0,5);t=lab.started
        lab.update([1,0],0,cfg['target'],False,t+1)
        lab.update([3.8,0],0,cfg['target'],False,t+2)
        self.assertEqual(lab.metrics()['status'],'arrived')
        self.assertAlmostEqual(lab.metrics()['path_length_m'],3.8)
        cfg['target'][0]=100
        self.assertEqual(lab.export()['config']['target'][0],4)
        lab.begin(cfg,[0,0],0,5);lab.update([0,0],0,cfg['target'],True,lab.started+1)
        self.assertEqual(lab.metrics()['collision_events'],1)
        self.assertEqual(lab.metrics()['status'],'collision')
        lab.begin(cfg,[0,0],0,5);t=lab.started
        lab.update([0,0],0,cfg['target'],False,t)
        lab.update([0,0],0,cfg['target'],False,t+4)
        self.assertEqual(lab.metrics()['stuck_events'],1)
        lab.update([0,0],0,cfg['target'],False,t+5)
        self.assertEqual(lab.metrics()['status'],'timeout')

if __name__=='__main__': unittest.main()

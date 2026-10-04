import unittest
from tool.simulator.observed_retreat_selection import shortlist, choose


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.robot = {'shape':'circle','radius':.2,'length':.4,'width':.3}
        self.cells = {(i,j):'clear' for i in range(-30,31) for j in range(-15,16)}
        self.plans = [{'id':f'retreat_{n}_{side}','stages':[{'linear_mps':-.3,'yaw_radps':0,'duration_s':n/30}]}
                      for side in ('left','right') for n in (60,120,180)]

    def test_reduce_six_retreats_to_two_without_claiming_model_preference(self):
        result = shortlist(self.plans,[0,0],0,self.robot,self.cells,.1)
        self.assertEqual([p['id'] for p in result['plans']],['retreat_180_left','retreat_180_right'])
        self.assertIsNone(result['model_preference'])
        self.assertEqual(choose(result,[{'strategy_id':'retreat_180_left'}])['id'],'retreat_180_right')

    def test_single_unknown_cell_excludes_long_sweep(self):
        del self.cells[(-17,0)]
        result = shortlist(self.plans,[0,0],0,self.robot,self.cells,.1)
        self.assertEqual([p['id'] for p in result['plans']],['retreat_120_left','retreat_120_right'])

    def test_blocked_current_footprint_refuses_every_retreat(self):
        self.cells[(0,0)] = 'blocked'
        result = shortlist(self.plans,[0,0],0,self.robot,self.cells,.1)
        self.assertEqual(result['plans'],[])
        self.assertIsNone(choose(result))


if __name__ == '__main__':
    unittest.main()

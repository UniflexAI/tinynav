import unittest
from types import SimpleNamespace
from tool.simulator.rule_recovery import RuleRecovery, recovery_candidates


class RuleTest(unittest.TestCase):
    def node(self):
        return SimpleNamespace(running=True,world_mode='observed',config={'target':[4,0,0],'robot':{'max_angular_vel':.75}},
            lab=SimpleNamespace(run={'status':'running'},metrics=lambda:{'elapsed_s':10},recent=lambda:{'window_s':8,'moved_m':0}),
            plan_received_at=10,collision=False,control_xy=[0,0],yaw_deg=0,override=None,override_until=0,
            recovery=SimpleNamespace(active=None,memory=[],settle_until=0),experiment_events=[])

    def test_unique_clear_progress_pivot_prevents_unnecessary_retreat(self):
        robot={'length':.4,'width':.3}
        cells={(i,j):'clear' for i in range(-30,31) for j in range(-20,21)}
        def plan(identity,v):
            return {'id':identity,'predicted_goal_progress_m':.2,'stages':[{'linear_mps':v,'yaw_radps':0,'duration_s':2}]}
        right=plan('pivot_right',.3);retreat=plan('retreat_60_left',-.3)
        selected=recovery_candidates([right,retreat],[0,0],0,robot,cells,.1)
        self.assertEqual(selected['plans'],[right])
        left=plan('pivot_left',.3)
        selected=recovery_candidates([right,left,retreat],[0,0],0,robot,cells,.1)
        self.assertEqual(selected['plans'],[retreat])

    def test_observed_pivot_fallback_requires_measured_sweep(self):
        robot={'length':.4,'width':.3}
        cells={(i,j):'clear' for i in range(-30,31) for j in range(-20,21)}
        plans=[{'id':'pivot_'+side,'predicted_goal_progress_m':progress,'stages':[{'linear_mps':.3,'yaw_radps':0,'duration_s':2}]} for side,progress in [('left',.2),('right',.3)]]
        result=recovery_candidates(plans,[0,0],0,robot,cells,.1)
        self.assertEqual(result['plans'][0]['id'],'pivot_right')
        cells.clear()
        self.assertEqual(recovery_candidates(plans,[0,0],0,robot,cells,.1)['plans'],[])

    def test_scan_budget_includes_scans_without_execution(self):
        node=self.node();p=RuleRecovery();p.scans=3;p.step(node,10.1)
        self.assertEqual(p.phase,'attempt_limit');self.assertIsNone(p.scan)

    def test_stale_report_cannot_start_scan(self):
        node=self.node();p=RuleRecovery();p.step(node,11)
        self.assertIsNone(p.scan);self.assertIsNone(node.override)

    def test_cancel_releases_scan_and_does_not_execute(self):
        node=self.node();p=RuleRecovery();p.step(node,10.1)
        self.assertEqual(p.phase,'scanning');self.assertEqual(node.override['linear_mps'],0)
        node.running=False;p.step(node,10.2)
        self.assertIsNone(p.scan);self.assertIsNone(node.override);self.assertIsNone(node.recovery.active)

    def test_stale_report_aborts_active_scan(self):
        node=self.node();p=RuleRecovery();p.step(node,10.1);p.step(node,11)
        self.assertEqual(p.phase,'scan_aborted');self.assertIsNone(node.override)

    def test_drift_aborts_active_scan(self):
        node=self.node();p=RuleRecovery();p.step(node,10.1);node.control_xy=[.03,0];p.step(node,10.2)
        self.assertEqual(p.phase,'scan_aborted');self.assertIsNone(node.override)

    def test_completed_scan_waits_for_new_report(self):
        node=self.node();p=RuleRecovery();p.step(node,10.1);p.scan['rotation_deg']=360
        p.step(node,10.2);self.assertEqual(p.phase,'fresh_report');self.assertEqual(node.override['yaw_radps'],0)
        p.step(node,10.3);self.assertIsNone(p.last_selection);self.assertIsNone(node.recovery.active)

    def test_full_scene_cannot_start_scan(self):
        node=self.node();node.world_mode='full_scene';p=RuleRecovery();p.step(node,10.1)
        self.assertEqual(p.phase,'unsupported_scene');self.assertIsNone(node.override)


if __name__=='__main__':unittest.main()

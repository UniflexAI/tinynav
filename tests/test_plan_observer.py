import unittest
from tool.simulator.decision_observer import planning_questions, validate_response, QUESTIONS

class PlanObserverTests(unittest.TestCase):
    def test_only_finite_ungated_alternatives_offered(self):
        state={'planning':{'selected_id':1,'top_candidates':[
            {'id':1,'reasons':[]}, {'id':2,'reasons':['reverse_gate_penalty']}, {'id':3,'reasons':[]}]}}
        q=planning_questions(state)
        self.assertIn('planning_issue',q)
        self.assertEqual(set(q['alternative']['criteria']),{'keep_current','request_new_candidates','uncertain','candidate_3'})

    def test_out_of_report_choice_rejected(self):
        questions=planning_questions({'planning':{'selected_id':1,'top_candidates':[]}})
        answers={key:{'choice':next(iter(spec['criteria'])),'probabilities':{k:float(i==0) for i,k in enumerate(spec['criteria'])}} for key,spec in questions.items() if spec['type']=='choice'}
        answers['stuck']={'noul':0.0}
        validate_response({'answers':answers},questions)
        answers['alternative']['choice']='candidate_999'
        with self.assertRaises(ValueError):validate_response({'answers':answers},questions)

if __name__=='__main__':unittest.main()

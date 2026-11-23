import ast
import copy
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

def candidate(name,robust,nominal,motion):
    return dict(meta=dict(candidate_id=name,task_quality=dict(nominal=dict(gravity_feasible=True,minimum_task_margin=nominal),robust=dict(minimum_task_margin=robust)),quality_terms=dict(planned_motion_s=motion,min_table_clearance_m=.003,uncertainty_m=.0015),quality_score=1.))

class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.ns={'np':np}
        functions(HERE/'stage5_control.py',{'s12_order_candidates'},self.ns)
        self.old=self.ns['s12_order_candidates']
        functions(HERE/'stage16_numerics_overlay.py',{'s16_order_candidates','s16_ik_terms'},self.ns)
    def test_normal_loss_per_contact_invariant(self):
        theta=.2
        def loss(n,w):return np.sum(np.full(n,2*(1-np.cos(theta)))*w*w*2/n)
        self.assertAlmostEqual(loss(2,4),loss(3,4))
        self.assertAlmostEqual(loss(3,7)/loss(2,4),49/16)
    def test_sensitivity_gap_is_not_tie_band(self):
        values=[candidate('two',.8,1.8,2),candidate('three',1.2,1.5,4)]
        self.assertEqual(self.old(copy.deepcopy(values))[0]['meta']['candidate_id'],'two')
        self.assertEqual(self.ns['s16_order_candidates'](copy.deepcopy(values))[0]['meta']['candidate_id'],'three')
    def test_true_relative_tie_keeps_shorter_motion(self):
        values=[candidate('two',1.15,9.,2),candidate('three',1.2,1.5,4)]
        self.assertEqual(self.ns['s16_order_candidates'](values)[0]['meta']['candidate_id'],'two')
    def test_nominal_gap_does_not_change_robust_order(self):
        for nominal in (1.2,20.,200.):
            values=[candidate('two',.8,nominal,2),candidate('three',1.2,1.5,4)]
            self.assertEqual(self.ns['s16_order_candidates'](values)[0]['meta']['candidate_id'],'three')
    def test_unsupported_fallback_does_not_outrank_supported(self):
        a=candidate('supported',.1,.2,4);b=candidate('fallback',20.,20.,1)
        b['meta']['task_quality']['nominal']['gravity_feasible']=False
        self.assertEqual(self.ns['s16_order_candidates']([a,b])[0]['meta']['candidate_id'],'supported')
    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):self.ns['s16_order_candidates']([], -.1)

if __name__=='__main__':unittest.main()

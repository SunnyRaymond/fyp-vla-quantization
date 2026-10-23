import unittest
from core import rho_step,aggregate

C={'minimum_rho':0.03,'maximum_rho':3,'probe_factor':1.5,'outer_step_factor':1.5,'relative_deadband':0.001}

class Controls(unittest.TestCase):
    def test_direction_and_deadband(self):
        self.assertAlmostEqual(rho_step(1.5,.04,.03,C)['rho_after'],2.25)
        self.assertAlmostEqual(rho_step(1.5,.03,.04,C)['rho_after'],1)
        self.assertAlmostEqual(rho_step(1.5,.03,.030001,C)['rho_after'],1.5)
    def test_bounds(self):
        x=rho_step(3,.04,.03,C)
        self.assertAlmostEqual(x['rho_plus'],3);self.assertAlmostEqual(x['rho_after'],3)
        self.assertAlmostEqual(rho_step(.03,.03,.04,C)['rho_after'],.03)
    def test_complete_paired_and_suite_mean(self):
        suites=('libero_spatial','libero_object','libero_goal','libero_10')
        rows=[{'case_id':c,'seed_index':s,'suite':suites[c],'m':float(c+s)} for c in range(4) for s in (0,1)]
        self.assertEqual(aggregate(rows,list(range(4)),[0,1],('m',))['overall']['m'],2)
        with self.assertRaises(ValueError):aggregate(rows[:-1],list(range(4)),[0,1],('m',))
        with self.assertRaises(ValueError):aggregate(rows+[rows[0]],list(range(4)),[0,1],('m',))

if __name__=='__main__':unittest.main()

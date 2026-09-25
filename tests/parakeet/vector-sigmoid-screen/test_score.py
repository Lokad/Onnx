from fractions import Fraction as F
import unittest
from score import ORDER,evaluate
from census import census


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cases=census()['cases']
    def totals(self,ratio=F(1,4)):
        return {n:[F(1) if n.startswith('current') else ratio]*46 for n in ORDER}
    def test_exact_prediction_boundary_passes(self):self.assertTrue(evaluate(self.totals(),self.cases)['admitted'])
    def test_less_than_prediction_fails(self):self.assertFalse(evaluate(self.totals(F(251,1000)),self.cases)['admitted'])
    def test_fallback_regression_cannot_hide_in_weighted_sum(self):
        t=self.totals();t[ORDER[1]][-1]=t[ORDER[2]][-1]=F(1051,1000)
        self.assertFalse(evaluate(t,self.cases)['admitted'])
    def test_failed_candidate_repeatability_rejects(self):
        t=self.totals();t[ORDER[1]][0]=F(1,8)
        self.assertFalse(evaluate(t,self.cases)['admitted'])
    def test_failed_current_repeatability_rejects(self):
        t=self.totals();t[ORDER[0]][0]=F(111,100)
        self.assertFalse(evaluate(t,self.cases)['admitted'])
    def test_actual_frequency_census(self):
        self.assertEqual(1920,sum(c['weight'] for c in self.cases))
        self.assertEqual(38,sum(c['weight']>0 for c in self.cases))
        self.assertEqual(2,sum(c['weight']==96 for c in self.cases))


if __name__=='__main__':unittest.main()

import copy,unittest
from audit import performance

def fixture():
    return [dict(order=i,bank=[dict(m=m,mode=mode,samples_ms=[100 if mode<2 else 97 if m in [30,128] else 100]*7) for m in [8,30,128,512] for mode in range(3)]) for i in range(1,5)]

class ScreenTests(unittest.TestCase):
    def test_pass_and_failed_gain(self):
        good=fixture();self.assertTrue(performance(good)['nominated'])
        for r in good:
            next(v for v in r['bank'] if v['m']==128 and v['mode']==2)['samples_ms']=[99]*7
        result=performance(good);self.assertTrue(result['controls_pass']);self.assertFalse(result['nominated'])
    def test_duplicate_instability(self):
        good=fixture();next(v for v in good[0]['bank'] if v['m']==128 and v['mode']==1)['samples_ms']=[106]*7
        self.assertFalse(performance(good)['controls_pass'])
    def test_worker_regression(self):
        good=fixture();next(v for v in good[0]['bank'] if v['m']==30 and v['mode']==2)['samples_ms']=[103]*7
        self.assertFalse(performance(good)['primary_gain_pass'])
    def test_malformed_samples_and_coverage(self):
        for bad in ([1]*6,[1]*8,[0]*7,[-1]*7,[float('nan')]*7,[float('inf')]*7):
            good=fixture();good[0]['bank'][0]['samples_ms']=bad
            with self.assertRaises(AssertionError):performance(good)
        good=fixture();good[0]['bank'][1]=copy.deepcopy(good[0]['bank'][0])
        with self.assertRaises(AssertionError):performance(good)
        with self.assertRaises(AssertionError):performance(fixture()[:3])

if __name__=='__main__':unittest.main()

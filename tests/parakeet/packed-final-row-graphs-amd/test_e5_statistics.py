import copy,unittest
from statistics_e5 import summarize

def value(ticks):return dict(mode='timing',calls=1380,clocks=[dict(index=i,warmup=i<1200,ticks=ticks,frequency=1000) for i in range(1380)])
def rows():return {key:value(10 if key.startswith('ort') else 20) for key in ['current-a','candidate-a','ort-a','ort-b','candidate-b','current-b']}

class Gates(unittest.TestCase):
    def test_ratio(self):self.assertEqual(summarize(rows())['ratio'],2)
    def test_boundary(self):
        r=rows();r['current-b']=value(22);self.assertTrue(summarize(r)['qualified'])
    def test_bad_control(self):
        r=rows();r['ort-b']=value(12);self.assertFalse(summarize(r)['qualified'])
    def test_candidate_control(self):
        r=rows();r['candidate-b']=value(23);self.assertFalse(summarize(r)['qualified'])
    def test_regression_boundary(self):
        r=rows();r['candidate-a']=r['candidate-b']=value(21)
        self.assertTrue(summarize(r)['qualified'])
    def test_regression_rejected_despite_stability(self):
        r=rows();r['candidate-a']=r['candidate-b']=value(22)
        result=summarize(r)
        self.assertTrue(all(c['passed'] for c in result['controls']))
        self.assertFalse(result['qualified'])
    def test_warmup_not_scored(self):
        r=rows()
        for c in r['current-a']['clocks'][:1200]:c['ticks']=100000
        self.assertEqual(summarize(r)['ratio'],2)
    def test_missing_clock(self):
        r=rows();r['current-a']['clocks'].pop()
        with self.assertRaises(AssertionError):summarize(r)
    def test_reordered_clock(self):
        r=rows();r['current-a']['clocks'][1200]['index']=61
        with self.assertRaises(AssertionError):summarize(r)
    def test_hidden_exclusion(self):
        r=rows();r['current-a']['clocks'][1200]['warmup']=True
        with self.assertRaises(AssertionError):summarize(r)
    def test_invalid_clock(self):
        for field,value_ in [('ticks',0),('frequency',0),('ticks',float('nan'))]:
            r=rows();r['current-a']['clocks'][1200][field]=value_
            with self.assertRaises(AssertionError):summarize(r)
    def test_last_clock_and_equal_process_weights(self):
        r=rows();r['current-a']['clocks'][-1]['ticks']+=1800
        result=summarize(r)
        self.assertEqual(result['means']['current-a'],.03)
        self.assertEqual(result['current'],.025)
        self.assertEqual(result['candidate_over_current'],.8)
    def test_wrong_declared_count(self):
        r=rows();r['current-a']['calls']=120
        with self.assertRaises(AssertionError):summarize(r)
    def test_just_above_repeatability_boundary(self):
        r=rows();r['current-b']=value(22001)
        for c in r['current-b']['clocks']:c['frequency']=1000000
        self.assertFalse(summarize(r)['qualified'])
    def test_just_above_regression_boundary(self):
        r=rows()
        for role in ['candidate-a','candidate-b']:
            r[role]=value(21001)
            for c in r[role]['clocks']:c['frequency']=1000000
        result=summarize(r)
        self.assertTrue(all(c['passed'] for c in result['controls']))
        self.assertFalse(result['qualified'])

if __name__=='__main__':unittest.main()

import copy,unittest
from statistics import summarize

def value(ticks):return dict(mode='timing',calls=120,clocks=[dict(index=i,warmup=i<60,ticks=ticks,frequency=1000) for i in range(120)])
def rows():return {key:value(20 if key.startswith('current') else 10) for key in ['current-a','ort-a','ort-b','current-b']}

class Gates(unittest.TestCase):
    def test_ratio(self):self.assertEqual(summarize(rows())['ratio'],2)
    def test_boundary(self):
        r=rows();r['current-b']=value(22);self.assertTrue(summarize(r)['qualified'])
    def test_bad_control(self):
        r=rows();r['ort-b']=value(12);self.assertFalse(summarize(r)['qualified'])
    def test_warmup_not_scored(self):
        r=rows()
        for c in r['current-a']['clocks'][:60]:c['ticks']=100000
        self.assertEqual(summarize(r)['ratio'],2)
    def test_missing_clock(self):
        r=rows();r['current-a']['clocks'].pop()
        with self.assertRaises(AssertionError):summarize(r)
    def test_reordered_clock(self):
        r=rows();r['current-a']['clocks'][60]['index']=61
        with self.assertRaises(AssertionError):summarize(r)
    def test_hidden_exclusion(self):
        r=rows();r['current-a']['clocks'][60]['warmup']=True
        with self.assertRaises(AssertionError):summarize(r)
    def test_invalid_clock(self):
        for field,value_ in [('ticks',0),('frequency',0),('ticks',float('nan'))]:
            r=rows();r['current-a']['clocks'][60][field]=value_
            with self.assertRaises(AssertionError):summarize(r)

if __name__=='__main__':unittest.main()

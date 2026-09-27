import copy
import unittest
from audit import analyze


class TimingAuditTests(unittest.TestCase):
    def setUp(self):
        self.spec=dict(shapes=[dict(m=m,n=1024,k=k,control=k==224) for m in [1024,2048] for k in [222,225,224]],
            jobs=[dict(name=n,role=r) for n,r in [('current-1','baseline'),('candidate-1','candidate'),('candidate-2','candidate'),('current-2','baseline')]],
            measurements=5,repeatability_limit=1.10,regression_limit=1.05)
        self.times={}
        for job in self.spec['jobs']:
            weights={222:10,225:8,224:8} if job['role']=='baseline' else {222:6,225:7,224:8}
            self.times[job['name']]=[[weights[s['k']]]*5 for s in self.spec['shapes']]

    def test_predicted_gain_requires_repeatability_and_controls(self):
        self.assertTrue(analyze(self.times,self.spec)['component_admitted'])
        wrong=copy.deepcopy(self.times);wrong['candidate-1'][0][0]*=1.2
        result=analyze(wrong,self.spec);self.assertFalse(result['component_admitted']);self.assertTrue(any(not r['passed'] for r in result['controls']))
        wrong=copy.deepcopy(self.times)
        for n in ['candidate-1','candidate-2']:wrong[n][2]=[9]*5
        self.assertFalse(analyze(wrong,self.spec)['component_admitted'])

    def test_missing_clock_or_failed_prediction_cannot_pass(self):
        wrong=copy.deepcopy(self.times);wrong['current-1'][0].pop()
        with self.assertRaises(AssertionError):analyze(wrong,self.spec)
        wrong=copy.deepcopy(self.times)
        for n in ['candidate-1','candidate-2']:
            wrong[n][0]=[8]*5;wrong[n][1]=[5]*5
        result=analyze(wrong,self.spec)
        self.assertFalse(result['component_admitted']);self.assertFalse(result['contrasts'][0]['passed'])


if __name__=='__main__':unittest.main()

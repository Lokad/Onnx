"""Adversarial checks of the frozen denominator, gates and retained clock census."""
import copy
from fractions import Fraction as F
from pathlib import Path
import unittest
from fixtures import screen_cases
from protocol import ORDER, read
from score import evaluate, score

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT/'artifacts/parakeet-scalar-where-layout-amd-20260923/collected/capture/result.json'
REFERENCE = ROOT/'artifacts/parakeet-provider-where-numerics-amd-v2-20260924/collected/current-numerics-256/result.json'


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cases = screen_cases(read(CAPTURE), read(REFERENCE))

    def totals(self, target=F(9,10)):
        return {name: [target if name.startswith('candidate') and c['partition']=='target6' else F(1)
                       for c in self.cases] for name in ORDER}

    def reports(self):
        results={}
        for sequence,name in enumerate(ORDER):
            rows=[]
            for i,c in enumerate(self.cases):
                values=1
                for d in c['output_shape']:values*=d
                rows.append(dict(index=i,name=c['name'],dtype=c['dtype'],batch=c['batch'],values=values,shape=c['output_shape'],
                    output=c['expected_output'],exact=True,inputs=True,owned=True,held=True,
                    clocks=[dict(index=i,name=c['name'],iteration=j,warmup=j<60,batch=c['batch'],ticks=1000*c['batch']) for j in range(120)]))
            results[name]=dict(completed=True,protocol='parakeet-provider-where-complete-call-60-60-v1',sequence=sequence,
                               role=name.split('-')[0],frequency=1000000,rows=rows)
        return results

    def test_exact_ten_percent_boundary(self):
        self.assertTrue(evaluate(self.totals(),self.cases)['admitted'])
        self.assertFalse(evaluate(self.totals(F(900001,1000000)),self.cases)['admitted'])

    def test_fallback_gain_cannot_supply_target(self):
        totals=self.totals(F(1))
        for name in ORDER[1:3]:totals[name]=[F(1) if c['partition']=='target6' else F(1,2) for c in self.cases]
        self.assertFalse(evaluate(totals,self.cases)['admitted'])

    def test_five_percent_case_boundary(self):
        totals=self.totals()
        for name in ORDER[1:3]:totals[name][0]=F(21,20)
        self.assertTrue(evaluate(totals,self.cases)['admitted'])
        for name in ORDER[1:3]:totals[name][0]+=F(1,1000000)
        self.assertFalse(evaluate(totals,self.cases)['admitted'])

    def test_case_repeatability(self):
        totals=self.totals();totals[ORDER[3]][0]=F(1200001,1000000)
        result=evaluate(totals,self.cases)
        self.assertTrue(any(not c['passed'] and c['scope']==self.cases[0]['name'] for c in result['controls']))

    def test_aggregate_repeatability(self):
        totals=self.totals();totals[ORDER[3]]=[v*F(1100001,1000000) for v in totals[ORDER[3]]]
        result=evaluate(totals,self.cases)
        self.assertTrue(any(not c['passed'] and c['scope']=='all122' for c in result['controls']))

    def test_strict_separation(self):
        totals=self.totals()
        for i,c in enumerate(self.cases):
            if c['partition']=='target6':totals[ORDER[1]][i]=F(1)
        gate=evaluate(totals,self.cases)['gates'][0]
        self.assertFalse(gate['passed'])

    def test_warmups_excluded(self):
        reports=self.reports();before=score(reports,self.cases)
        for report in reports.values():
            for row in report['rows']:
                for clock in row['clocks'][:60]:clock['ticks']*=100000
        self.assertEqual(score(reports,self.cases),before)

    def test_single_measured_tick_preserved_and_batch_denominator(self):
        reports=self.reports();before=score(reports,self.cases)['rows'][0]['current']
        reports[ORDER[0]]['rows'][0]['clocks'][-1]['ticks']+=1
        after=score(reports,self.cases)['rows'][0]['current']
        delta=F(after['numerator'],after['denominator'])-F(before['numerator'],before['denominator'])
        self.assertEqual(delta,F(1,2*60*1000000*self.cases[0]['batch']))

    def test_zero_clock_and_deleted_clock_refused(self):
        reports=self.reports();reports[ORDER[0]]['rows'][0]['clocks'][0]['ticks']=0
        with self.assertRaises(AssertionError):score(reports,self.cases)
        reports=self.reports();reports[ORDER[0]]['rows'][0]['clocks'].pop()
        with self.assertRaises(AssertionError):score(reports,self.cases)

    def test_wrong_output_or_batch_refused(self):
        for key,value in [('output','0'*64),('batch',1)]:
            reports=self.reports();reports[ORDER[0]]['rows'][0][key]=value
            with self.assertRaises(AssertionError):score(reports,self.cases)

    def test_census_and_process_order_refused(self):
        reports=self.reports();reports[ORDER[0]]['rows'].pop()
        with self.assertRaises(AssertionError):score(reports,self.cases)
        reports=self.reports()
        with self.assertRaises(AssertionError):score(dict(reversed(list(reports.items()))),self.cases)


if __name__=='__main__':unittest.main()

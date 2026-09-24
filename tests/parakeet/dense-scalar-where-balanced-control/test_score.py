"""Adversarial checks for false improvements, repeatability and complete clocks."""
from fractions import Fraction as F
from pathlib import Path
import unittest
from fixtures import screen_cases
from protocol import ORDER, read
from score import evaluate, score

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = ROOT/'artifacts/parakeet-scalar-where-layout-amd-20260923/collected/capture/result.json'
REFERENCE = ROOT/'artifacts/parakeet-dense-scalar-where-numerics-amd-20260924/collected/current-numerics-256/result.json'


class Scoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cases = screen_cases(read(CAPTURE), read(REFERENCE))

    def totals(self): return {name: [F(1) for _ in self.cases] for name in ORDER}

    def reports(self):
        results={}
        for sequence,name in enumerate(ORDER):
            rows=[]
            for i,c in enumerate(self.cases):
                values=1
                for d in c['output_shape']:values*=d
                rows.append(dict(index=i,name=c['name'],dtype=c['dtype'],batch=c['batch'],values=values,shape=c['output_shape'],
                    output=c['expected_output'],exact=True,inputs=True,owned=True,held=True,
                    clocks=[dict(index=i,name=c['name'],iteration=j,warmup=j<600,batch=c['batch'],ticks=1000*c['batch']) for j in range(780)]))
            results[name]=dict(completed=True,protocol='parakeet-dense-where-balanced-600-180-v1',sequence=sequence,
                               role=name.split('-')[0],frequency=1000000,rows=rows)
        return results

    def test_identical_means_pass(self):
        self.assertTrue(evaluate(self.totals(),self.cases)['admitted'])

    def test_symmetric_five_percent_case_boundaries(self):
        for boundary,delta in [(F(21,20),F(1,1000000)),(F(20,21),-F(1,1000000))]:
            totals=self.totals()
            for name in ORDER[1:3]:totals[name][0]=boundary
            self.assertTrue(evaluate(totals,self.cases)['admitted'])
            for name in ORDER[1:3]:totals[name][0]+=delta
            self.assertFalse(evaluate(totals,self.cases)['admitted'])

    def test_middle_speedup_is_false_positive(self):
        totals=self.totals()
        for name in ORDER[1:3]:totals[name]=[v*F(9,10) for v in totals[name]]
        result=evaluate(totals,self.cases)
        self.assertFalse(result['admitted']);self.assertFalse(result['gates'][1]['passed'])

    def test_all_four_process_case_repeatability(self):
        totals=self.totals();totals[ORDER[0]][0]=F(6,5)
        result=evaluate(totals,self.cases)
        self.assertTrue(next(c for c in result['controls'] if c['scope']==self.cases[0]['name'])['passed'])
        totals[ORDER[0]][0]+=F(1,1000000)
        self.assertFalse(next(c for c in evaluate(totals,self.cases)['controls'] if c['scope']==self.cases[0]['name'])['passed'])

    def test_all_four_aggregate_repeatability(self):
        totals=self.totals();totals[ORDER[3]]=[F(11,10) for _ in self.cases]
        self.assertTrue(next(c for c in evaluate(totals,self.cases)['controls'] if c['scope']=='all220')['passed'])
        totals[ORDER[3]]=[v+F(1,1000000) for v in totals[ORDER[3]]]
        self.assertFalse(next(c for c in evaluate(totals,self.cases)['controls'] if c['scope']=='all220')['passed'])

    def test_added_cases_cannot_be_hidden(self):
        totals=self.totals()
        for name in ORDER[1:3]:totals[name][-1]=F(106,100)
        self.assertFalse(evaluate(totals,self.cases)['admitted'])

    def test_warmups_excluded(self):
        reports=self.reports();before=score(reports,self.cases)
        for report in reports.values():
            for row in report['rows']:
                for clock in row['clocks'][:600]:clock['ticks']*=100000
        self.assertEqual(score(reports,self.cases),before)

    def test_every_measured_tick_and_batch_denominator(self):
        reports=self.reports();before=score(reports,self.cases)['rows'][0]['outer']
        reports[ORDER[0]]['rows'][0]['clocks'][-1]['ticks']+=1
        after=score(reports,self.cases)['rows'][0]['outer']
        delta=F(after['numerator'],after['denominator'])-F(before['numerator'],before['denominator'])
        self.assertEqual(delta,F(1,2*180*1000000*self.cases[0]['batch']))

    def test_zero_deleted_or_relabelled_clock_refused(self):
        reports=self.reports();clocks=reports[ORDER[0]]['rows'][0]['clocks'];first=clocks[0]['ticks'];clocks[0]['ticks']=0
        with self.assertRaises(AssertionError):score(reports,self.cases)
        clocks[0]['ticks']=first;last=clocks.pop()
        with self.assertRaises(AssertionError):score(reports,self.cases)
        clocks.append(last);clocks[600]['warmup']=True
        with self.assertRaises(AssertionError):score(reports,self.cases)

    def test_wrong_output_or_batch_refused(self):
        reports=self.reports();row=reports[ORDER[0]]['rows'][0];original=row['output'];row['output']='0'*64
        with self.assertRaises(AssertionError):score(reports,self.cases)
        row['output']=original;row['batch']+=1
        with self.assertRaises(AssertionError):score(reports,self.cases)

    def test_census_and_process_order_refused(self):
        reports=self.reports();last=reports[ORDER[0]]['rows'].pop()
        with self.assertRaises(AssertionError):score(reports,self.cases)
        reports[ORDER[0]]['rows'].append(last)
        with self.assertRaises(AssertionError):score(dict(reversed(list(reports.items()))),self.cases)


if __name__=='__main__':unittest.main()

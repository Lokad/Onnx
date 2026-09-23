import copy
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
from score import ORDER,SHORT,LONG,evaluate,score


class Gates(unittest.TestCase):
    def totals(self):
        return {name:[F(4,5) if name.startswith('candidate') and i in SHORT else F(1) for i in range(21)] for name in ORDER}
    def test_pass(self):self.assertTrue(evaluate(self.totals())['admitted'])
    def test_ten_percent_boundary(self):
        t=self.totals()
        for name in ORDER[1:3]:
            for i in SHORT:t[name][i]=F(9,10)
        self.assertTrue(evaluate(t)['admitted'])
        for name in ORDER[1:3]:t[name][0]+=F(1,10**12)
        self.assertFalse(evaluate(t)['admitted'])
    def test_long_controls_cannot_supply_short_gain(self):
        t=self.totals()
        for name in ORDER[1:3]:t[name]=[F(1) if i in SHORT else F(1,10) for i in range(21)]
        self.assertFalse(evaluate(t)['admitted'])
    def test_control_regression(self):
        t=self.totals()
        for name in ORDER[1:3]:t[name][3]=F(1050001,1000000)
        self.assertFalse(evaluate(t)['admitted'])
    def test_corpus_repeat(self):
        t=self.totals();t[ORDER[3]]=[F(111,100)]*21
        self.assertFalse(evaluate(t)['admitted'])
    def test_case_repeat(self):
        t=self.totals();t[ORDER[3]][0]=F(121,100)
        self.assertFalse(evaluate(t)['admitted'])
    def test_strict_process_separation(self):
        t=self.totals();t[ORDER[1]]=[F(3,5)]*21;t[ORDER[2]]=[F(1)]*21
        result=evaluate(t);self.assertFalse(result['admitted'])
        self.assertFalse(next(g for g in result['gates'] if g['name']=='strict-separation-short12')['passed'])


class Clocks(unittest.TestCase):
    def data(self):
        capture=json.loads((Path(__file__).resolve().parents[3]/'artifacts/parakeet-short-dispatch-numerics-amd-v2-20260923/bundle/fixtures/result.json').read_text())
        reports={}
        for sequence,name in enumerate(ORDER):
            role=name.split('-')[0];rows=[]
            for i,f in enumerate(capture['entries']):
                ticks=80 if role=='candidate' and i in SHORT else 100
                rows.append(dict(index=i,name=f['name'],node=f['node'],m=f['m'],reduction=f['k'],columns=f['n'],preparationTicks=1,
                    exact=True,guards=True,inputs=True,output=f['y']['sha256'],clocks=[dict(iteration=j,warmup=j<60,ticks=ticks) for j in range(120)]))
            reports[name]=dict(passed=True,protocol='parakeet-short-wide-complete-call-60-60-v1',sequence=sequence,role=role,
                calls=2520,warmups=1260,measured=1260,frequency=100,rows=rows)
        return reports,capture
    def test_complete_census_and_weighting(self):
        r=score(*self.data());self.assertTrue(r['admitted']);self.assertEqual(r['scopes']['short12']['current']['value'],12)
        self.assertEqual(r['scopes']['all21']['candidate']['value'],18.6)
    def test_warmups_excluded(self):
        reports,capture=self.data();reports[ORDER[0]]['rows'][0]['clocks'][0]['ticks']=10**15
        self.assertEqual(score(reports,capture),score(*self.data()))
    def test_one_measured_tick_retained(self):
        reports,capture=self.data();reports[ORDER[0]]['rows'][0]['clocks'][60]['ticks']+=1
        r=score(reports,capture)['rows'][0]['current']
        self.assertEqual(F(r['numerator'],r['denominator']),1+F(1,12000))
    def test_bad_census_clock_or_output_rejected(self):
        for change in ['missing','duplicate','zero','fractional','hash','shape','order']:
            reports,capture=self.data();row=reports[ORDER[0]]['rows'][0]
            if change=='missing':row['clocks'].pop()
            elif change=='duplicate':row['clocks'][1]=copy.deepcopy(row['clocks'][0])
            elif change=='zero':row['clocks'][0]['ticks']=0
            elif change=='fractional':row['clocks'][0]['ticks']=.1
            elif change=='hash':row['output']='0'*64
            elif change=='shape':row['m']+=1
            else:reports=dict(reversed(list(reports.items())))
            with self.assertRaises(AssertionError):score(reports,capture)


if __name__=='__main__':unittest.main()

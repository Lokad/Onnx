"""Exercise exact thresholds and reject unstable or selectively regressed corpora."""
import copy
import unittest
from checks import evaluate
from statistics_exact import timing_table
from protocol import TIMING_ROLES

def fixture(candidate=970):
    cases=[dict(name=f'clip-{i}',samples=1 if i<19 else 3412240-19) for i in range(20)]
    results=[]
    for role in TIMING_ROLES:
        ticks=1000 if role=='current' else 800 if role=='ort' else candidate
        results.append(dict(records=[dict(name=case['name'],phase='warmup' if p==0 else 'measured',start_ticks=0,end_ticks=ticks,frequency=1000) for case in cases for p in range(4)]))
    return results,dict(cases=cases)

class Admission(unittest.TestCase):
    def test_exact_three_percent_boundary(self):
        results,manifest=fixture();value=evaluate(timing_table(results,manifest))
        self.assertTrue(value['admitted']);self.assertEqual(len(value['controls']),63)
        self.assertFalse(value['parity_target_met'])
        results,manifest=fixture(971);self.assertFalse(evaluate(timing_table(results,manifest))['admitted'])

    def test_per_clip_boundary_even_with_large_total_gain(self):
        for ticks,expected in [(1050,True),(1051,False)]:
            results,manifest=fixture(800)
            for i in [1,4]:
                for row in results[i]['records']:
                    if row['name']=='clip-0':row['end_ticks']=ticks
            self.assertEqual(evaluate(timing_table(results,manifest))['admitted'],expected)

    def test_each_role_repeatability(self):
        for index in [0,1,2]:
            results,manifest=fixture(700)
            for row in results[index]['records']:row['end_ticks']*=2
            value=evaluate(timing_table(results,manifest))
            self.assertFalse(value['admitted']);self.assertFalse(value['controls_passed'])

    def test_warmups_not_scored_but_all_measures_retained(self):
        results,manifest=fixture();expected=timing_table(results,manifest)
        for result in results:
            for row in result['records']:
                if row['phase']=='warmup':row['end_ticks']=999999999
        self.assertEqual(timing_table(results,manifest),expected)
        results[0]['records'].pop()
        with self.assertRaises(AssertionError):timing_table(results,manifest)

    def test_corpus_must_equal_all_clip_means(self):
        results,manifest=fixture();table=timing_table(results,manifest)
        row=table[-1]['candidate'];row['exact_mean']=dict(numerator=1,denominator=1)
        for p in row['processes']:p['exact_mean']=dict(numerator=1,denominator=1)
        with self.assertRaises(AssertionError):evaluate(table)

if __name__=='__main__':unittest.main()

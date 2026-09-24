"""Reject unstable, insufficient, or regressing timings without using warmup clocks."""
import unittest
from checks import evaluate
from protocol import TIMING_JOBS


def workers(candidate=80):
    cases=['a','b','c','d','e','f'];values={}
    for name in TIMING_JOBS:
        ticks=candidate if name.startswith('candidate') else 100
        values[name]=dict(frequency=1000,rows=[dict(name=case,phase=phase,repeat=repeat,ticks=ticks if phase=='measured' else 1000000)
            for phase in ['warmup','measured'] for repeat in range(5) for case in cases])
    return values,cases


class Decisions(unittest.TestCase):
    def test_stable_gain_passes_with_exact_control_census(self):
        values,cases=workers();result=evaluate(values,cases)
        self.assertTrue(result['admitted']);self.assertEqual(84,len(result['controls']));self.assertEqual(14,len(result['gates']))
        self.assertEqual(dict(numerator=4,denominator=5),result['table'][0]['candidate_over_selected'])

    def test_small_gain_is_valid_but_rejected(self):
        values,cases=workers(95);result=evaluate(values,cases)
        self.assertTrue(result['controls_passed']);self.assertFalse(result['admitted'])

    def test_single_process_repeat_spike_invalidates_apparent_gain(self):
        values,cases=workers()
        for row in values['selected-0-512']['rows']:
            if row['phase']=='measured' and row['repeat']==0:row['ticks']*=2
        result=evaluate(values,cases);self.assertFalse(result['controls_passed']);self.assertFalse(result['admitted'])

    def test_case_regression_rejects_large_corpus_gain(self):
        values,cases=workers(60)
        for name,result in values.items():
            if name.startswith('candidate'):
                for row in result['rows']:
                    if row['name']=='a' and row['phase']=='measured':row['ticks']=106
        result=evaluate(values,cases);self.assertTrue(result['controls_passed']);self.assertFalse(result['admitted'])

    def test_duplicate_process_mismatch_invalidates(self):
        values,cases=workers()
        for row in values['selected-1-256']['rows']:
            if row['phase']=='measured':row['ticks']=130
        self.assertFalse(evaluate(values,cases)['controls_passed'])

    def test_warmup_clocks_do_not_change_decision(self):
        values,cases=workers();before=evaluate(values,cases)
        for result in values.values():
            for row in result['rows']:
                if row['phase']=='warmup':row['ticks']=1
        self.assertEqual(before,evaluate(values,cases))


if __name__=='__main__':unittest.main()

"""Reject unstable, mislabelled and unexpectedly reconstructed diagnostic clocks."""
from fractions import Fraction
import unittest
from measurements import JOBS,repeatability,seconds,summarize


def fixture():
    frames=[167]*7+[120]*13
    cases=[dict(name=f'clip-{i}',expected=dict(encoded_frames=n)) for i,n in enumerate(frames)]
    clocks={job:[dict(name=case['name'],phase='warmup' if p==0 else 'measured',
                     encoder_ticks=2000000000,encoder_frequency=1000000000,profile_frequency=10000000,
                     feed_forward_ticks=10000000,math_ticks=7000000,scale_ticks=1000000,
                     copy_y_ticks=1000000 if job.startswith('candidate') and i<7 else 0)
                for p in range(4) for i,case in enumerate(cases)] for job in JOBS}
    return clocks,cases


class Tests(unittest.TestCase):
    def test_equal_runs_and_all_controls(self):
        clocks,cases=fixture();value=summarize(clocks,cases)
        self.assertTrue(value['usable_for_attribution']);self.assertEqual(len(value['controls']),92)
        self.assertEqual(value['table'][-1]['candidate']['copy_y']['seconds'],.7)

    def test_distinct_clock_units(self):
        row=fixture()[0][JOBS[1]][20]
        self.assertEqual(seconds(row,'encoder'),Fraction(2))
        self.assertEqual(seconds(row,'copy_y'),Fraction(1,10))

    def test_corpus_boundary_is_exact(self):
        self.assertTrue(repeatability(Fraction(1),Fraction(11,10),True)['passed'])
        self.assertFalse(repeatability(Fraction(1),Fraction(11000001,10000000),True)['passed'])

    def test_clip_bound_is_distinct(self):
        self.assertTrue(repeatability(Fraction(1),Fraction(6,5),False)['passed'])
        self.assertFalse(repeatability(Fraction(1),Fraction(12000001,10000000),False)['passed'])

    def test_unstable_copy_does_not_pass_with_stable_encoder(self):
        clocks,cases=fixture()
        for row in clocks[JOBS[2]]:row['copy_y_ticks']*=2
        value=summarize(clocks,cases)
        self.assertFalse(value['usable_for_attribution'])
        self.assertEqual(sum(not r['passed'] for r in value['controls']),8)

    def test_unexpected_copy_is_rejected(self):
        clocks,cases=fixture();clocks[JOBS[0]][20]['copy_y_ticks']=1
        with self.assertRaises(AssertionError):summarize(clocks,cases)

    def test_missing_clip_is_rejected(self):
        clocks,cases=fixture();clocks[JOBS[0]].pop()
        with self.assertRaises(AssertionError):summarize(clocks,cases)

    def test_warmup_cannot_be_relabelled(self):
        clocks,cases=fixture();clocks[JOBS[0]][0]['phase']='measured'
        with self.assertRaises(AssertionError):summarize(clocks,cases)


if __name__=='__main__':unittest.main()

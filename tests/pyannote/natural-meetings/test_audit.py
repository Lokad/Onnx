import copy
import unittest
from audit import inspect_public, compare


def fixture():
    return dict(status='Completed',windows=591,audio_seconds=600,intervals=[[1.,3.,0],[2.,4.,1]],exclusive_intervals=[[1.,3.,0],[3.,4.,1]],
                speakers=[dict(speaker=i,centroid=[.01]*256,has_embedding=True) for i in range(2)])


class PublicAuditTests(unittest.TestCase):
    def test_malformed_outputs_are_refused(self):
        inspect_public(fixture(),9600000)
        mutations=[lambda v:v.update(windows=592),lambda v:v.update(audio_seconds=601),lambda v:v['speakers'].pop(),
                   lambda v:v['speakers'][0]['centroid'].pop(),lambda v:v['speakers'][0]['centroid'].__setitem__(0,float('nan')),
                   lambda v:v['intervals'][0].__setitem__(0,-1),lambda v:v['intervals'][0].__setitem__(2,.5),
                   lambda v:v['exclusive_intervals'][1].__setitem__(0,2),lambda v:v.update(status='NoSpeech')]
        for mutate in mutations:
            value=fixture();mutate(value)
            with self.assertRaises(AssertionError):inspect_public(value,9600000)

    def test_disagreements_are_retained_without_weakening_gates(self):
        original=fixture();self.assertTrue(compare(original,original)['passed'])
        altered=copy.deepcopy(original);altered['speakers'][0]['centroid'][0]+=.00011
        self.assertFalse(compare(altered,original)['passed'])
        altered=copy.deepcopy(original);altered['intervals'][0][1]+=1e-10
        self.assertFalse(compare(altered,original)['passed'])
        altered=copy.deepcopy(original);altered['intervals'].pop()
        self.assertEqual(compare(altered,original)['mismatches'][0]['reason'],'length')
        altered=copy.deepcopy(original);altered['speakers'][0]['centroid'][0]+=.00009
        self.assertTrue(compare(altered,original)['passed'])


if __name__=='__main__':unittest.main()

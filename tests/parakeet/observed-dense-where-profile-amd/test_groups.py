"""Check attribution against retained real profiles and reject missing/changed work."""
import copy
from functools import lru_cache
import unittest
from run import ROOT,PRIOR,MATCHED,read
from groups import compare_where


@lru_cache(maxsize=1)
def fixture():
    phase=read(PRIOR/'analysis.json')['phases']['wall']
    matched=read(MATCHED/'analysis.json')
    return phase,matched,ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'


class GroupTests(unittest.TestCase):
    def test_retained_totals_and_shared_ancestors(self):
        phase,matched,model=fixture();result=compare_where(phase,phase,matched,model)
        expected=sum(r['managed_seconds'] for r in matched['rows'] if 'pad' not in r['kind'])
        self.assertEqual(result['kernels'],72);self.assertAlmostEqual(result['selected_seconds'],expected,places=12)
        self.assertEqual(result['gain'],0);self.assertEqual(result['complete_group_gain'],0)
        self.assertLess(result['complete_groups']['selected']['seconds'],sum(r['totals']['selected']['seconds'] for r in result['families']))
        self.assertTrue(any(r['members']>1 for r in result['union_rows']))
        for family in result['families']:
            expected=matched['families'][family['kind']]['managed']
            self.assertEqual(family['totals']['selected'],expected)

    def test_missing_node(self):
        phase,matched,model=fixture();candidate=copy.deepcopy(phase)
        candidate['node_rows']=[r for r in candidate['node_rows'] if r['name']!='/layers.0/self_attn/Where']
        with self.assertRaises(AssertionError):compare_where(phase,candidate,matched,model)

    def test_changed_calls_or_output(self):
        phase,matched,model=fixture()
        for key,value in [('calls',59),('outputs',['different-output'])]:
            candidate=copy.deepcopy(phase)
            row=next(r for r in candidate['node_rows'] if r['name']=='/layers.0/self_attn/Where')
            row[key]=value
            with self.assertRaises(AssertionError):compare_where(phase,candidate,matched,model)

    def test_duplicate_or_changed_boundary(self):
        phase,matched,model=fixture()
        for mode in ['duplicate','boundary','scalar']:
            modified=copy.deepcopy(matched)
            if mode=='duplicate':modified['rows'][1]=copy.deepcopy(modified['rows'][0])
            elif mode=='boundary':modified['rows'][0]['boundary']=['different-mask','different-data']
            else:modified['rows'][0]['true_scalar_bits']='00000000'
            with self.assertRaises(AssertionError):compare_where(phase,phase,modified,model)


if __name__=='__main__':unittest.main()

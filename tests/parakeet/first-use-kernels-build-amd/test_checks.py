"""Reject collateral metadata changes, including the similarly named double kernel."""
import copy
import unittest
from checks import METHODS, flags


def fixture():
    keys=['Lokad.Onnx.MathOps::'+name+'::Void '+name+'(Single*)' for name in METHODS]
    double='Lokad.Onnx.MathOps::mm_unsafe_vectorized_intrinsics::Void mm_unsafe_vectorized_intrinsics(Double*)'
    before={key:0 for key in [*keys,double,'unrelated']}
    after=dict(before)
    for key in keys:after[key]=512
    return dict(assembly='Lokad.Onnx.dll',method_flags_before=before,method_flags_after=after,compiler_rename=None),keys,double


class MetadataChecks(unittest.TestCase):
    def test_exact_four_flags(self):
        row,keys,_=fixture();self.assertEqual(flags(row,True),keys)

    def test_reject_collateral_flags(self):
        for kind in ['double','unrelated','missing','extra','wrong_bit','removed']:
            with self.subTest(kind=kind):
                row,keys,double=fixture();after=row['method_flags_after']
                if kind=='double':after[double]=512
                if kind=='unrelated':after['unrelated']=8
                if kind=='missing':after[keys[0]]=0
                if kind=='extra':after['new_method']=0
                if kind=='wrong_bit':after[keys[0]]=256
                if kind=='removed':del after['unrelated']
                with self.assertRaises((AssertionError,KeyError)):flags(row,True)

    def test_no_data_change(self):
        row=dict(assembly='Lokad.Onnx.Data.dll',method_flags_before={'data':0},method_flags_after={'data':0},compiler_rename=None)
        self.assertEqual(flags(row,True),[])
        row['method_flags_after']['data']=512
        with self.assertRaises(AssertionError):flags(row,True)

    def test_current_wrapper_helpers_and_compiler_name(self):
        row,keys,_=fixture();before=row['method_flags_before'];after=row['method_flags_after']
        before.update(wrapper=0,old_lambda=0);after.update(wrapper=256,new_lambda=0,helper1=8,helper2=8)
        row.update(differences=['wrapper'],added=['helper1','helper2'],compiler_rename=dict(oldKey='old_lambda',newKey='new_lambda'))
        self.assertEqual(flags(row),keys+['wrapper'])
        broken=copy.deepcopy(row);broken['method_flags_after']['helper1']=0
        with self.assertRaises(AssertionError):flags(broken)


if __name__=='__main__':unittest.main()

"""Reject unrelated compiled changes and malformed added-helper inventories."""
from copy import deepcopy
import unittest
from checks import ADDED,CHANGED,inventory


class Scope(unittest.TestCase):
    def test_scope_mutations(self):
        def key(pair):return '::'.join(pair)+'::signature'
        changes=list(map(key,sorted(CHANGED)));added=list(map(key,sorted(ADDED)))
        product={n:dict(sha256=n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        rows=[]
        for name,count in [('Lokad.Onnx.dll',3163),('Lokad.Onnx.Data.dll',697)]:
            iscore=name=='Lokad.Onnx.dll';different=changes if iscore else [];extra=added if iscore else []
            methods={k:'before' for k in different}
            methods.update({str(i):'unchanged' for i in range(count-len(methods))})
            rows.append(dict(assembly=name,methods=count,normalized_methods=methods,public_surface_equal=True,compiler_rename=None,
                before_sha256=name,after_sha256=name,removed=[],differences=different,added=extra,
                unchanged_methods=count-len(different),candidate_methods={k:'after' for k in different+extra}))
        value=dict(inventory_complete=True,observations=rows);inventory(value,product,product)
        changesets=[lambda v:v.update(inventory_complete=False),lambda v:v['observations'][0].update(public_surface_equal=False),
            lambda v:v['observations'][0].update(compiler_rename={}),lambda v:v['observations'][0].update(removed=['unexpected']),
            lambda v:v['observations'][0]['differences'].append('unexpected::method::signature'),
            lambda v:v['observations'][0]['added'].append('unexpected::method::signature'),
            lambda v:v['observations'][0]['candidate_methods'].update({added[0]:'FusedMultiplyAdd'}),
            lambda v:v['observations'][0]['candidate_methods'].update({added[0]:'NO-BODY'}),
            lambda v:v['observations'][0]['candidate_methods'].update({changes[0]:'before'}),
            lambda v:v['observations'][0].update(unchanged_methods=3160),
            lambda v:v['observations'][1].update(differences=['unexpected']),
            lambda v:v['observations'][1].update(added=['unexpected']),
            lambda v:v['observations'][0].update(before_sha256='wrong')]
        for mutate in changesets:
            damaged=deepcopy(value);mutate(damaged)
            with self.assertRaises(AssertionError):inventory(damaged,product,product)


if __name__=='__main__':unittest.main()

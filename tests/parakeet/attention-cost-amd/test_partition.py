"""Exercise rejection with real current weight/request/route metadata."""
import copy
import unittest
from partition import partition
from run import PROFILE, APP, read, attention_metadata, references, ROOT, PROVIDER, MATMUL, changed


def fixture():
    expected = attention_metadata()
    manifest = read(APP/'collected/manifests/current-parakeet.json')
    records = copy.deepcopy(read(PROFILE/'capture-collected/control/result.json')['records'])
    rows = []
    for i, request in enumerate(records):
        request.update(start_ticks=i*100000,end_ticks=i*100000+50000)
        for j, meta in enumerate(expected['rows'][i*120:(i+1)*120]):
            m, fallback = meta['m'], meta['route'] != 'mapped-allowed'
            main = (m if m%3 == 0 else m-m%2) if fallback else 0
            counts = [1,1,1,1,int(m != main)] if fallback else [0]*5
            start = request['start_ticks']+j*100+10
            starts = [start+k*10 for k in range(5)]
            rows.append(dict(index=len(rows),weight=meta['weight'],m=m,main_rows=main,
                mapped=meta['route']!='unmapped',row_guard_allows=m%2==0 or m%3==0,fallback=fallback,
                start_ticks=start,end_ticks=start+70,stage_calls=counts,
                stage_ticks=[5*c for c in counts],stage_starts=[t*c for t,c in zip(starts,counts)],
                stage_ends=[(t+5)*c for t,c in zip(starts,counts)],simd=True,intrinsics=True,degree=1,
                leaf=('ShortWideMultiply3Rows' if m%3==0 else 'ShortWideMultiply2Rows') if fallback else '',
                scratch_elements=1024**2 if fallback else 0,thread=1,input_layout='test',weight_layout='test'))
    costs = dict(passed=True,protocol='parakeet-attention-cost-v1',frequency=1000000000,runtime='10.0.8',
        processor_count=1,fma=True,avx2=True,avx512=True,flags={},
        stages=['rent','pack','arithmetic','return','final_row'],rows=rows)
    return costs, records, manifest, expected


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = fixture()

    def test_complete_current_coverage_retains_remainder(self):
        groups = partition(*self.args)
        total, = [r for r in groups if r['group']==['total']]
        self.assertEqual(total['measured_calls'],7200)
        for group in groups:
            self.assertAlmostEqual(sum(group['stages'].values())+group['remainder_seconds'],group['corpus_seconds'])
            self.assertGreater(group['remainder_seconds'],0)

    def test_reject_missing_wrong_route_and_overlapping_intervals(self):
        costs,*rest = self.args
        target = next(i for i,r in enumerate(costs['rows']) if r['fallback'])
        mutations = [lambda c:c['rows'].pop(),
            lambda c:c['rows'][1].update(weight=c['rows'][0]['weight']),
            lambda c:c['rows'][target].update(mapped=not c['rows'][target]['mapped']),
            lambda c:c['rows'][target].update(m=999),
            lambda c:c['rows'][target].update(leaf='unexpected'),
            lambda c:c['rows'][target].update(start_ticks=-1),
            lambda c:c['rows'][target]['stage_ticks'].__setitem__(1,999),
            lambda c:c['rows'][target]['stage_starts'].__setitem__(1,c['rows'][target]['stage_starts'][0]),
            lambda c:c['rows'][target]['stage_calls'].__setitem__(1,0),
            lambda c:c['rows'][target].update(thread=2)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value=copy.deepcopy(costs); mutate(value)
                with self.assertRaises(AssertionError): partition(value,*rest)

    def test_source_hooks_bind_current_qualified_product(self):
        source,_ = references()
        self.assertEqual(len(source),445)
        for name in [PROVIDER,MATMUL]:
            original=(ROOT/name).read_bytes()
            self.assertNotEqual(changed(name,original),original)


if __name__ == '__main__': unittest.main()

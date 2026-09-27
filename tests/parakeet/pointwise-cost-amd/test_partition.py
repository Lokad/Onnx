"""Reject incomplete, overlapping or misattributed pointwise observations."""
import copy
import unittest
from audit import partition


def fixture():
    records, rows = [], []
    manifest = dict(cases=[dict(name=str(i), expected=dict(encoded_frames=51+i)) for i in range(20)])
    for index in range(80):
        start = index*10000
        records.append(dict(name=str(index%20),phase='warmup' if index<20 else 'measured',start_ticks=start,end_ticks=start+5000))
        for offset, filters in enumerate([2048,1024]*24):
            tick = start+offset*100+10
            rows.append(dict(index=len(rows),filters=filters,columns=51+index%20,
                start_ticks=tick,end_ticks=tick+70,stage_calls=[1]*5,stage_ticks=[10]*5,
                simd=True,intrinsics=True,degree=1,leaf='mm_unsafe_vectorized_intrinsics_2x4packed_bump',
                scratch_elements=1024*128,thread=1,input_same=True,weight_same=True,
                input_layout='dense',weight_layout='dense'))
    costs = dict(passed=True,protocol='parakeet-pointwise-cost-v1',frequency=1000000000,runtime='10.0.8',
        processor_count=1,fma=True,avx2=True,avx512=True,flags={},
        stages=['output_initialize','materialize','clear','pack','arithmetic'],rows=rows)
    return costs, records, manifest


class PartitionTests(unittest.TestCase):
    def test_complete_accounting_keeps_remainder(self):
        groups = partition(*fixture())
        self.assertEqual([g['measured_calls'] for g in groups],[1440,1440])
        for group in groups:
            self.assertAlmostEqual(sum(group['stages'].values())+group['remainder_seconds'],group['corpus_seconds'])
            self.assertGreater(group['remainder_seconds'],0)

    def test_rejects_corrupt_coverage_or_attribution(self):
        original, records, manifest = fixture()
        mutations = [lambda c:c['rows'].pop(),
            lambda c:c['rows'][48].update(start_ticks=0),
            lambda c:c['rows'][100].update(stage_ticks=[100]*5),
            lambda c:c['rows'][100].update(stage_calls=[1,1,1,0,1]),
            lambda c:c['rows'][100].update(filters=256),
            lambda c:c['rows'][100].update(columns=999),
            lambda c:c['rows'][100].update(leaf='different-kernel')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                costs = copy.deepcopy(original)
                mutate(costs)
                with self.assertRaises(AssertionError): partition(costs,records,manifest)


if __name__ == '__main__': unittest.main()

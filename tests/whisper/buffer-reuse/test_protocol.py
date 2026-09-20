"""Check prospective allocation/cache gates at their boundaries and on damaged records."""
from pathlib import Path
import copy,sys,unittest
sys.path.append(str(Path(__file__).resolve().parents[1]/'memory-collection'))
from reuse_protocol import BUDGETS,validate_reuse,allocation_gate
from memory_protocol import INTEGER_FIELDS


def observation():
    def memory(t):
        row={k:0 for k in INTEGER_FIELDS};row.update(ticks=t,total_available_memory=16*1024**3,high_memory_threshold=15*1024**3,
            collections=[0,0,0],last_gc_concurrent=False,last_gc_compacted=False)
        return row
    return dict(diagnostic='bounded-context-reuse-no-forced-gc',records=[dict(name=str(i),input_sha256=str(i),result={},
        allocated_bytes=50,start_ticks=i*10+2,end_ticks=i*10+3,memory_before=memory(i*10+1),memory_after=memory(i*10+4),
        gc_before=[0,0,0],gc_after=[0,0,0],pools={name:dict(allocated_new_bytes=16*1024**2,reused_bytes=0,cache_bytes=budget,cache_count=256,cache_budget=budget)
            for name,budget in BUDGETS.items()}) for i in range(20)])


class ProtocolTests(unittest.TestCase):
    def test_accepts_exact_limits_and_does_not_require_collection_or_heap_drop(self):
        value=observation();validate_reuse(value)
        original=copy.deepcopy(value['records'][:16])
        for row in original:row['allocated_bytes']=100
        self.assertEqual(allocation_gate(value,original)['ratio'],.5)

    def test_rejects_damaged_cache_allocation_and_snapshot_records(self):
        for damage in [lambda v:v.update(diagnostic='forced-gc'),
            lambda v:v['records'][0]['pools']['encodingExecution'].update(cache_count=257),
            lambda v:v['records'][0]['pools']['encodingExecution'].update(cache_bytes=512*1024**2+1),
            lambda v:v['records'][0]['pools']['firstExecution'].update(cache_budget=0),
            lambda v:v['records'][1]['pools']['encodingExecution'].update(allocated_new_bytes=16*1024**2+1),
            lambda v:v['records'][0].update(allocated_bytes=-1),
            lambda v:v['records'][0].update(gc_after=[-1,0,0]),
            lambda v:v['records'][1]['memory_before'].update(ticks=1)]:
            value=observation();damage(value)
            with self.assertRaises(AssertionError):validate_reuse(value)

    def test_rejects_allocation_gate_mismatch_and_one_byte_over_limit(self):
        value=observation();original=copy.deepcopy(value['records'][:16])
        for row in original:row['allocated_bytes']=100
        for damage in [lambda v:v['records'][1].update(allocated_bytes=51),
            lambda v:v['records'][0].update(name='different'),lambda v:v['records'].pop()]:
            bad=copy.deepcopy(value);damage(bad)
            with self.assertRaises(AssertionError):allocation_gate(bad,original)


if __name__=='__main__':unittest.main()

"""Prospective allocation, state and cache gates; raw elapsed times are not benchmarks."""
from memory_protocol import transition

BUDGETS=dict(encodingExecution=512*1024**2,firstExecution=128*1024**2,pastExecution=128*1024**2)


def validate_reuse(value):
    assert value['diagnostic']=='bounded-context-reuse-no-forced-gc'
    previous=None
    for index,row in enumerate(value['records']):
        before=row['memory_before'];after=row['memory_after'];transition(before,after)
        if previous is not None:transition(previous,before)
        previous=after
        assert before['ticks']<=row['start_ticks']<row['end_ticks']<=after['ticks']
        assert type(row['allocated_bytes']) is int and row['allocated_bytes']>=0
        assert len(row['gc_before'])==len(row['gc_after'])==3
        assert all(type(n) is int and n>=0 for n in row['gc_before']+row['gc_after'])
        assert all(a<=b<=c<=d for a,b,c,d in zip(before['collections'],row['gc_before'],row['gc_after'],after['collections']))
        assert set(row['pools'])==set(BUDGETS)
        for name,budget in BUDGETS.items():
            p=row['pools'][name]
            assert set(p)=={'allocated_new_bytes','reused_bytes','cache_bytes','cache_count','cache_budget'}
            assert all(type(v) is int and v>=0 for v in p.values())
            assert p['cache_budget']==budget and p['cache_bytes']<=budget and p['cache_count']<=256
        if index>0:assert row['pools']['encodingExecution']['allocated_new_bytes']<=16*1024**2


def allocation_gate(value,original):
    assert len(value['records'])==20 and len(original)==16
    for old,new in zip(original,value['records']):
        for key in ['name','input_sha256','result']:assert old[key]==new[key],key
    before=sum(row['allocated_bytes'] for row in original[1:16])
    after=sum(row['allocated_bytes'] for row in value['records'][1:16])
    assert before>0 and after*2<=before
    return dict(passed=True,first_call=2,last_call=16,original_allocated_bytes=before,prototype_allocated_bytes=after,
                ratio=after/before,encoder_warm_max=max(row['pools']['encodingExecution']['allocated_new_bytes'] for row in value['records'][1:]))

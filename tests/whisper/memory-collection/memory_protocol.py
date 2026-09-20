"""Validate the fixed collection intervention without requiring memory to decrease."""
import math

INTEGER_FIELDS=('ticks','allocated_total','managed_estimate','rss','private_bytes','last_gc_index',
    'last_gc_generation','last_gc_heap','last_gc_fragmented','last_gc_committed','last_gc_memory_load',
    'total_available_memory','high_memory_threshold','last_gc_promoted','last_gc_pinned','last_gc_finalization_pending')


def snapshot(value):
    assert set(value)==set(INTEGER_FIELDS)|{'collections','last_gc_concurrent','last_gc_compacted'}
    assert all(type(value[k]) is int and value[k]>=0 for k in INTEGER_FIELDS)
    assert value['last_gc_generation'] in [0,1,2]
    assert value['ticks']>0 and value['total_available_memory']>0 and value['high_memory_threshold']>0
    assert len(value['collections'])==3 and all(type(n) is int and n>=0 for n in value['collections'])
    assert type(value['last_gc_concurrent']) is type(value['last_gc_compacted']) is bool


def transition(before,after):
    snapshot(before);snapshot(after)
    assert before['ticks']<=after['ticks'] and before['allocated_total']<=after['allocated_total']
    assert before['last_gc_index']<=after['last_gc_index']
    assert all(a<=b for a,b in zip(before['collections'],after['collections']))


def validate_memory(value):
    assert value['diagnostic']=='explicit-gen2-after-8-16-20'
    assert len(value['records'])==20 and [v['after_call'] for v in value['collections']]==[8,16,20]
    interventions={c['after_call']:c for c in value['collections']};previous=None
    for index,row in enumerate(value['records'],1):
        before=row['memory_before'];after=row['memory_after'];transition(before,after)
        if previous is not None:transition(previous,before)
        assert before['ticks']<=row['start_ticks']<row['end_ticks']<=after['ticks']
        assert all(a<=b<=c<=d for a,b,c,d in zip(before['collections'],row['gc_before'],row['gc_after'],after['collections']))
        previous=after
        if index in interventions:
            c=interventions[index];transition(previous,c['before']);transition(c['before'],c['after'])
            assert all(type(c[k]) is int for k in ['start_ticks','end_ticks','frequency']) and c['frequency']>0
            assert c['before']['ticks']<=c['start_ticks']<c['end_ticks']<=c['after']['ticks']
            assert math.isfinite(c['seconds']) and c['seconds']>0 and math.isclose(c['seconds'],(c['end_ticks']-c['start_ticks'])/c['frequency'],rel_tol=1e-14)
            assert c['held_outputs_unchanged'] is True and c['inputs_unchanged'] is True
            assert c['after']['collections'][2]>c['before']['collections'][2] and c['after']['last_gc_index']>c['before']['last_gc_index']
            previous=c['after']


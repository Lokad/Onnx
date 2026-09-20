"""Additional checks bind reported calls to frozen executables and raw evidence."""
import copy
import math
from common import pin, read
from audit import inspect_worker


def worker(value, manifest, frozen, base, engine):
    inspect_worker(value, manifest, engine)
    assert value['manifest_sha256'] == pin(base/'manifest.json')['sha256']
    assert value['held_outputs_unchanged'] is True
    for row in value['records']:
        assert row['ownership'] is True and math.isfinite(row['seconds']) and row['seconds'] > 0
    if engine == 'managed':
        assert value['runner_sha256'] == frozen['files']['bin/NaturalMeetings.dll']['sha256']
        for row in value['records']:
            assert type(row['allocated_bytes']) is int and row['allocated_bytes'] >= 0
            before, after = row['gc_before'], row['gc_after']
            assert len(before) == len(after) == 3
            assert all(type(a) is int and type(b) is int and 0 <= a <= b for a,b in zip(before,after))
    else:
        assert value['runner_sha256'] == frozen['files']['runtime/native.py']['sha256']
        assert value['adapter_sha256'] == manifest['native_sources']['tests/audio/comparison/native_adapters.py']['sha256']
        assert value['python_binary'] == frozen['native_files']['C:/Python313/python.exe']
        assert value['flags'] == dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def refuses(action):
    try:
        action()
    except (AssertionError, ValueError, KeyError, TypeError, IndexError):
        return
    raise AssertionError('Damaged evidence accepted')


def damaged_worker_checks(value, manifest, frozen, base, engine):
    changes = [lambda v:v['records'].pop(), lambda v:v['records'].reverse(),
               lambda v:v['records'][0].update(input_sha256='0'*64),
               lambda v:v.update(manifest_sha256='0'*64), lambda v:v.update(runner_sha256='0'*64),
               lambda v:v.update(affinity=1), lambda v:v.update(held_outputs_unchanged=1),
               lambda v:v['records'][0].update(ownership=1), lambda v:v['records'][0].update(frequency=0),
               lambda v:v['records'][0].update(end_ticks=v['records'][0]['start_ticks']),
               lambda v:v['records'][0]['result'].update(windows=590),
               lambda v:v['records'][0]['result']['speakers'][0]['centroid'].pop(),
               lambda v:v['records'][0]['result']['speakers'][0]['centroid'].__setitem__(0,float('nan')),
               lambda v:v['records'][0]['result']['intervals'][0].__setitem__(0,-1),
               lambda v:v['records'][0]['result']['intervals'][0].__setitem__(2,9999)]
    if engine=='managed':
        changes += [lambda v:v.update(core_sha256='0'*64), lambda v:v.update(data_sha256='0'*64),
                    lambda v:v['records'][0].update(allocated_bytes=-1),
                    lambda v:v['records'][0].update(gc_after=[-1,0,0])]
    else:
        changes += [lambda v:v.update(adapter_sha256='0'*64),lambda v:v.update(python_binary={}),
                    lambda v:v.update(flags={}),lambda v:v['native_settings'].update(intra_threads=2)]
    for change in changes:
        damaged=copy.deepcopy(value);change(damaged)
        refuses(lambda:worker(damaged,manifest,frozen,base,engine))
    return len(changes)


def resources(state, samples, frozen_sha256, limits, engine):
    assert state['complete'] is True and state['code'] == 0 and 'error' not in state
    assert state['engine'] == engine and state['mode'] == 'run'
    assert state['limits'] == limits and state['frozen_sha256'] == frozen_sha256
    assert 0 < state['seconds'] < limits['seconds'] and state['ended'] > state['started']
    assert type(state['samples']) is int and state['samples'] == len(samples) and samples
    assert state['supervisor']['birth'] <= state['child']['birth']
    assert state['members'][str(state['child']['pid'])] == state['child']['birth']
    seen = {}; previous = -1.; peak = 0
    for sample in samples:
        assert math.isfinite(sample['seconds']) and previous < sample['seconds'] < state['seconds']
        previous = sample['seconds']
        assert type(sample['available']) is int and sample['available'] >= limits['available']
        assert len({m['pid'] for m in sample['members']}) == len(sample['members'])
        for member in sample['members']:
            assert type(member['pid']) is int and member['pid'] > 0
            assert type(member['rss']) is int and member['rss'] >= 0
            assert member['affinity'] == [2]
            assert member['birth'] == state['members'][str(member['pid'])] >= state['child']['birth']
            seen[str(member['pid'])] = member['birth']
        rss = sum(m['rss'] for m in sample['members'])
        assert rss < limits['rss']; peak = max(peak, rss)
    assert peak == state['peak_rss'] and seen == state['members']
    assert state['accounting']['valid'] is True
    assert math.isfinite(state['accounting']['foreign_cpu_fraction'])
    assert state['accounting']['foreign_cpu_fraction'] >= 0


def damaged_resource_checks(state, samples, frozen_sha256, limits, engine):
    changes = [lambda s,x:s.update(complete=False), lambda s,x:s.update(code=1),
               lambda s,x:s.update(frozen_sha256='0'*64), lambda s,x:s.update(samples=len(x)+1),
               lambda s,x:s.update(seconds=limits['seconds']), lambda s,x:s.update(peak_rss=-1),
               lambda s,x:s['child'].update(birth=s['child']['birth']+1),
               lambda s,x:x[1].update(seconds=x[0]['seconds']),
               lambda s,x:x[0].update(available=limits['available']-1),
               lambda s,x:x[0]['members'][0].update(rss=-1),
               lambda s,x:x[0]['members'][0].update(rss=limits['rss']),
               lambda s,x:x[0]['members'][0].update(affinity=[0]),
               lambda s,x:x[0]['members'][0].update(birth=0),
               lambda s,x:s['accounting'].update(valid=False)]
    assert len(samples) > 1 and samples[0]['members']
    for change in changes:
        damaged_state, damaged_samples = copy.deepcopy((state, samples))
        change(damaged_state, damaged_samples)
        refuses(lambda:resources(damaged_state,damaged_samples,frozen_sha256,limits,engine))
    return len(changes)

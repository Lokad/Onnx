"""Independently reconcile the terminal no-model proof and both epoch bounds."""
import json
from pathlib import Path
from counter import EVENTS, epoch, intervals
from run import BASE, TOOLS, pin, read, save


def main():
    assert not (BASE/'closed.json').exists()
    prepared = read(BASE/'prepared.json')
    for name, wanted in prepared['files'].items(): assert pin(TOOLS/name) == wanted, name
    receipt = read(BASE/'collected.json'); assert receipt['terminal']
    folder = BASE/'collected'
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    for name in ['counter.py', 'proof.py']: assert pin(folder/name) == prepared['files'][name]
    state = read(folder/'state.json')
    assert state['owner'] == receipt['owner'] == read(BASE/'deployment.json')
    assert state['complete'] and state['terminal'] and state['code'] == 0
    assert state['inference_calls'] == 0 and not state['system_settings_changed']
    assert state['ended']-state['started'] < 30 and state['exitcodes'] == dict(frequency=0, workload=0)
    clock = state['runtime_clock']; assert len(clock['brackets']) == 1000
    assert clock['symbol'] == 'SystemNative_GetTimestamp' and clock['library'].endswith('/10.0.8/libSystem.Native.so')
    assert all(r['before_ns'] <= r['native_ns'] <= r['after_ns'] for r in clock['brackets'])
    raw = (folder/'counters/intervals.csv').read_text()
    rows = intervals(raw); assert rows == state['intervals']
    bound = epoch(state['anchors']); assert bound == state['epoch']
    for anchor in state['anchors']:
        chunk = raw.encode()[anchor['output_before_bytes']:anchor['output_after_bytes']].decode()
        single = intervals(chunk)
        assert len(single) == 1 and single[0]['elapsed_ns'] == anchor['elapsed_ns']
    work = read(folder/'workload.json'); assert state['workload'] == work
    assert 2300000000 <= work['end_ns']-work['start_ns'] < 3000000000
    full = []
    for before, after in zip(rows, rows[1:]):
        if bound['lower_ns']+before['elapsed_ns'] >= work['start_ns'] and bound['upper_ns']+after['elapsed_ns'] <= work['end_ns']:
            full.append(after)
    assert full and [r['elapsed_ns'] for r in full] == state['whole_workload_intervals']
    for row in full:
        for name in EVENTS[:3]:
            assert int(row['events'][name]['count']) > 0
            assert float(row['events'][name]['running_percent']) >= 99.9
    assert state['samples']
    for sample in state['samples']:
        assert sample['seconds'] < 30 and sample['rss'] == sum(m['rss'] for m in sample['members']) < 512*1024**2
        assert sample['available'] >= 1024**3 and sample['tmpfs'] >= 1024**3 and sample['output'] < 64*1024**2
        for m in sample['members']:
            assert state['identities'][str(m['pid'])] == m['birth']
            assert m['affinity'] == [2] if m['role'] == 'workload' else m['affinity'] in [[0], [2]]
    analysis = dict(passed=True, epoch=bound, anchors=state['anchors'], intervals=len(rows),
        whole_workload_intervals=state['whole_workload_intervals'], runtime_clock_library=clock['identity'],
        timestamp_brackets=1000, samples=len(state['samples']), peak_rss=max(r['rss'] for r in state['samples']),
        inference_calls=0, system_settings_changed=False)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, analysis=pin(BASE/'analysis.json'),
        local_inputs=prepared['files'], files=files, terminal_owner=state['owner'], inference_calls=0))
    print(json.dumps(dict(closure=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()

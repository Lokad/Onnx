"""Close the full actual-shape result, preserving every numerical failure."""
import json
from transform import transform
from build import BASE, FIXTURES, pin, read, save, verify, monitor


def main():
    assert not (BASE/'closed.json').exists()
    verified = read(BASE/'verified.json'); assert verified['complete']; verify(verified['files'])
    state = read(BASE/'controller.json'); assert state['complete'] and state['code'] in [0, 1]
    assert [r['name'] for r in state['runs']] == ['restore', 'build', 'qualify-256']
    identities = [state['supervisor']]; resources = []
    for row in state['runs']:
        raw = row['name'] == 'qualify-256'
        assert row['complete'] and row['code'] in ([0, 1] if raw else [0]) and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if raw else 8)*1024**3
        samples = [json.loads(s) for s in (BASE/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3
            assert sample['disk'] >= 20*1024**3 and sample['output_bytes'] <= 1024**3
            assert sample['rss'] == sum(m['rss'] for m in sample['members']) and (not raw or len(sample['members']) <= 1)
            for member in sample['members']: assert member['affinity'] == [2] and row['members'][str(member['pid'])] == member['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: monitor.terminal(identity)
    result = read(BASE/'output/256.json'); assert pin(BASE/'output/256.json') == verified['report']
    assert result['cases'] == len(result['observations']) == 108 and result['eligible'] == 96 and result['fallback'] == 12
    assert result['values'] == sum(r['values'] for r in result['observations'])
    assert result['differences'] == sum(r['differences'] for r in result['observations'])
    assert result['native_failures'] == sum(r['native_failed'] for r in result['observations'])
    assert result['maximum'] == max(r['maximum'] for r in result['observations'])
    assert result['failed'] == sum(r['differences'] != 0 or r['native_failed'] != 0 for r in result['observations'])
    assert result['passed'] == verified['passed'] == (result['failed'] == 0) == (state['code'] == 0)
    fixtures = read(FIXTURES/'output/result.json')
    for row, call in zip(result['observations'], fixtures['calls']):
        assert (row['name'], row['index'], row['node'], row['form'], row['eligible']) == (call['case'], call['index'], call['node'], call['form'], call['eligible'])
        assert row['repeats_and_ownership']
    original = read(BASE.parent/'pyannote-blocked-spatial-model-20260922/output/256.json')
    assert result['observations'] == original['observations']
    source = BASE.parent/'pyannote-blocked-spatial-model-20260922/source'
    for name in ['ModelProbe.cs', 'GeneratedKernels.cs']: assert pin(BASE/'source'/name) == pin(source/name)
    observed, diff = transform((source/'BlockedSpatial.cs').read_text())
    assert (BASE/'source/BlockedSpatial.cs').read_text() == observed and (BASE/'source.diff').read_text() == diff
    assert result['read_only_operands'] and result['prepared_weights'] == 32 and result['lanes'] == 8 and not result['flags']
    assert result['pid'] == state['runs'][-1]['worker']['pid']
    assert result['core'] == verified['core']['sha256'] and result['executable'] == verified['consumer']['sha256']
    analysis = dict(passed=result['passed'], qualification_complete=True, cases=108, values=result['values'], differences=result['differences'],
        native_failures=result['native_failures'], maximum=result['maximum'], failed=result['failed'], resources=resources, no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj', 'packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json', dict(passed=result['passed'], qualification_complete=True, files=files, local_inputs=verified['files'], identities=identities, analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()

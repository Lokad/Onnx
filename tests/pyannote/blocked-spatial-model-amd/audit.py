"""Require every actual AMD observation to match closed Windows qualification."""
import json
from run import BASE, prepared
from protocol import check_sample, pin, read, save


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    payload = read(BASE/'payload/payload.json'); collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == ['qualify-256', 'qualify-512']
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    original = read(BASE/'payload/windows-256.json'); resources = []; reports = {}
    fixtures = read(BASE/'payload/fixtures/result.json')
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= 12*1024**3 and row['preflight']['tmpfs'] >= 3*1024**3
        assert row['preflight'] == row['preflight_observations'][-1] == read(collected/(row['name']+'-preflight.json'))[-1]
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']: assert row['members'][str(member['pid'])] == member['birth']
        result = read(collected/row['name']/'result.json')
        assert result['passed'] and result['cases'] == len(result['observations']) == 108 and result['eligible'] == 96 and result['fallback'] == 12
        assert result['failed'] == result['differences'] == result['native_failures'] == 0
        assert result['observations'] == original['observations'] and result['values'] == original['values'] == 119823360
        assert result['maximum'] == original['maximum'] <= 1e-4
        assert result['prepared_weights'] == 32 and result['prepared_bytes'] == original['prepared_bytes']
        assert result['read_only_operands'] and result['runtime'] == '10.0.8' and result['avx512'] and not result['flags']
        assert result['lanes'] == (8 if row['name'] == 'qualify-256' else 16)
        assert result['core'] == payload['core']['sha256'] and result['executable'] == payload['consumer']['sha256'] and result['pid'] == row['child']['pid']
        for observed, call in zip(result['observations'], fixtures['calls']):
            assert (observed['name'], observed['index'], observed['form'], observed['eligible']) == (call['case'], call['index'], call['form'], call['eligible'])
            assert observed['production'] == observed['candidate'] and observed['repeats_and_ownership']
        reports[row['name']] = {k: result[k] for k in ['cases', 'values', 'maximum', 'lanes', 'avx512', 'runtime', 'prepared_bytes']}
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    analysis = dict(passed=True, reports=reports, resources=resources, samples=sum(r['samples'] for r in resources),
        peak_rss=max(r['peak_rss'] for r in resources), no_performance_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()

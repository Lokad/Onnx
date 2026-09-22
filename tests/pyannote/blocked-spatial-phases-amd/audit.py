"""Close numerical/resource evidence and independently evaluate all fixed gates."""
import json
import sys
from run import BASE, ROOT, prepared
from protocol import check_sample, pin, read, save
sys.path.insert(0, str(ROOT/'tests/pyannote/blocked-spatial-phases'))
from reconcile import analyze
ORDER = ['profile-a', 'profile-b']


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    tests = read(BASE/'selftest.json'); assert tests['passed'] and tests['tests'] == 8
    payload = read(BASE/'payload/payload.json'); collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0 and state['ended']-state['started'] < 4*3600
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == ['qualify-512', *ORDER]
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    original = read(BASE/'payload/windows-256.json'); resources = []; reports = {}
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < 900
        assert row['preflight']['available'] >= 12*1024**3 and row['preflight']['tmpfs'] >= 3*1024**3
        assert row['preflight'] == row['preflight_observations'][-1] == read(collected/(row['name']+'-preflight.json'))[-1]
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']: assert row['members'][str(member['pid'])] == member['birth']
        result = read(collected/row['name']/'result.json'); assert result['passed']
        assert result['read_only_operands'] and result['runtime'] == '10.0.8' and result['lanes'] == 16 and not result['flags']
        assert result['core'] == payload['core']['sha256'] and result['executable'] == payload['consumer']['sha256'] and result['pid'] == row['child']['pid']
        if row['name'] == 'qualify-512':
            assert result['observations'] == original['observations'] and result['values'] == 119823360
            assert result['failed'] == result['differences'] == result['native_failures'] == 0 and result['maximum'] <= 1e-4 and result['avx512']
        else:
            journal = [json.loads(s) for s in (collected/row['name']/'journal.jsonl').read_text().splitlines()]
            assert journal == result['rows']
            reports[row['name']] = result
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    analysis = analyze(reports, read(BASE/'payload/fixtures/result.json'), original)
    analysis.update(complete=True, numerical_and_resource_checks_pass=True, resources=resources,
        samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        no_application_speed_measurement=True, no_ort_speed_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, diagnostic_only=True, files=files, local_inputs=spec['files'],
        remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json'), scope='Complete phase reconciliation and numerical/resource evidence only; no speed selection.'))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), calls=analysis['calls'], totals={k:v['total'] for k,v in analysis['processes'].items()}, samples=analysis['samples'], peak_rss=analysis['peak_rss'])))


if __name__ == '__main__': main()

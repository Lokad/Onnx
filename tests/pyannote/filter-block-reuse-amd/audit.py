"""Independently reconcile every numerical result, resource sample and terminal owner."""
import json
from run import BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import check_result


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    payload = read(BASE/'payload/payload.json'); collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    for name, wanted in payload['files'].items(): assert pin(BASE/'payload'/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS
    assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources = []; reports = {}
    for row in state['runs']:
        mode, width = row['name'].split('-')
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        assert row['preflight'] == row['preflight_observations'][-1] == read(collected/(row['name']+'-preflight.json'))[-1]
        samples = [json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']: assert row['members'][str(member['pid'])] == member['birth']
        result = read(collected/row['name']/'result.json')
        assert result['pid'] == row['child']['pid'] and result['runtime'] == '10.0.8'
        reports[row['name']] = check_result(result, mode, width, payload, BASE/'payload')
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    assert state['ended']-state['started'] < 4*3600
    analysis = dict(passed=True, reports=reports, resources=resources,
        samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        core=payload['core'], consumers=payload['consumers'], no_performance_measurement=True, codegen_pending=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, files=files, local_inputs=spec['files'], remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), **analysis)))


if __name__ == '__main__': main()

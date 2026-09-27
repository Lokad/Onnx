"""Retain the complete isolated correctness verdict; never infer performance."""
import json
from pathlib import PurePosixPath
from protocol import JOBS, LIMITS, pin, read, save, check_sample
from prepare import BASE, previous_closed
from run import prepared, REMOTE
from commands import command_for
from checks import compiled, contracts


def main():
    assert not (BASE/'analysis.json').exists() and not (BASE/'closed.json').exists() and not (BASE/'failed.json').exists()
    previous_closed(); prepared()
    folder = BASE/'collected'
    receipt, payload, state = read(folder/'collection.json'), read(BASE/'payload.json'), read(folder/'identity.json')
    assert receipt['terminal'] and receipt['input_error'] is None
    assert receipt['excluded_regenerable_directories'] == ['http-cache']
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['receipt'] == pin(folder/'collection.json')
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    assert receipt['payload'] == pin(BASE/'payload.json') == pin(folder/'payload.json')
    assert state['complete'] and state['code'] == receipt['code'] and state['boot_time'] == 1789634288.0
    assert state['supervisor'] == read(BASE/'deployment.json')
    assert state['ended']-state['started'] < 4*3600
    assert payload['jobs'] == JOBS and payload['limits'] == LIMITS
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
    if state['code'] != 0:
        files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
        save(BASE/'failed.json', dict(passed=False, terminal=True, evidence_verified=True,
            state=state, performance_admitted=False, files=files))
        raise AssertionError('Original terminal failure retained; inspect it before a new namespace.')
    assert [r['name'] for r in state['runs']] == JOBS
    resources = []
    for row in state['runs']:
        assert row['complete'] and row['code'] == 0 and row['seconds'] < LIMITS['seconds']
        assert row['exitcodes'] == {'worker': 0} and set(row['processes']) == {'worker'}
        expected, _, cpu = command_for(PurePosixPath(REMOTE), row['name'], payload)
        assert row['commands'] == {'worker': list(map(str, expected))}
        owner = row['processes']['worker']
        assert owner['affinity'] == [cpu] and row['members'][str(owner['pid'])] == owner['birth']
        assert row['preflight']['available'] >= LIMITS['preflight_available'] and row['preflight']['tmpfs'] >= LIMITS['preflight_tmpfs']
        samples = [json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert row['members'][str(member['pid'])] == member['birth']
                assert row['affinities'][str(member['pid'])] == member['expected_affinity']
        assert max(s['rss'] for s in samples) == row['peak_rss']
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    built = read(folder/'built.json'); assert built['passed']
    for name, wanted in built['files'].items(): assert pin(folder/name) == wanted, name
    inventory = compiled(read(folder/'inventory/instructions.json'), payload['current_product'], built['candidate'])
    assert inventory == read(folder/'inventory/review.json')
    reports = {}
    for row in state['runs']:
        if row['name'].startswith('contracts-'):
            reports[row['name']] = contracts(folder/row['name'], payload, built, row)
            assert reports[row['name']] == read(folder/row['name']/'review.json')
    values = [r['projections']['hashes'] for r in reports.values()]
    assert len(values) == 3 and values[0] == values[1] == values[2]
    warnings = [line.strip() for p in (folder/'logs').glob('*.stdout')
        for line in p.read_text().splitlines() if ': warning ' in line]
    assert len(warnings) == 4 and all('warning CS8604:' in s and 'Zzz.WideProjectionEntry.cs(20,' in s for s in warnings)
    analysis = dict(passed=True, product_changed_in_isolation=True, root_product_changed=False,
        performance_admitted=False, baseline=payload['current_product'], candidate=built['candidate'],
        consumer=built['consumer'], compiled=inventory, contracts=reports, resources=resources,
        warnings=warnings, source=read(BASE/'bundle/stage.json')['source'])
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, remote_terminal=True, performance_admitted=False,
        analysis=pin(BASE/'analysis.json'), files=files))
    print(json.dumps(dict(passed=True, closed=pin(BASE/'closed.json'), candidate=built['candidate'],
        tests=sum(r['passed_tests'] for r in reports.values()), projection_values=3*1945600,
        resource_samples=sum(r['samples'] for r in resources), performance_admitted=False)))


if __name__ == '__main__': main()

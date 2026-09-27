"""Close complete build/contract evidence without making a performance claim."""
import json
from pathlib import PurePosixPath
from protocol import JOBS, LIMITS, pin, read, save, check_sample
from prepare import BASE, FIRST, SECOND, previous_closed
from run import prepared, REMOTE
from commands import command_for
from compiled import review


def main():
    assert not (BASE/'analysis.json').exists() and not (BASE/'closed.json').exists()
    previous_closed(); prepared()
    folder = BASE/'collected'
    receipt, payload, state = read(folder/'collection.json'), read(BASE/'payload.json'), read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert receipt['excluded_regenerable_directories'] == ['http-cache']
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['receipt'] == pin(folder/'collection.json')
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    assert receipt['payload'] == pin(BASE/'payload.json') == pin(folder/'payload.json')
    assert state['complete'] and state['code'] == 0 and state['boot_time'] == 1789634288.0
    assert state['supervisor'] == read(BASE/'deployment.json')
    assert state['ended']-state['started'] < 4*3600
    assert [r['name'] for r in state['runs']] == payload['jobs'] == JOBS and payload['limits'] == LIMITS
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p,b in r['members'].items()]
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
    prior_folder = SECOND/'collected'
    prior_state = read(prior_folder/'identity.json')
    assert pin(folder/'built.json') == pin(prior_folder/'built.json') == pin(FIRST/'collected/built.json')
    compiled = review(read(prior_folder/'inventory/instructions.json'), built, payload['current_product'])
    assert compiled == read(prior_folder/'inventory/review.json')
    reports = {}
    for mode in ['normal', 'noavx512', 'scalar']:
        data_folder, owner_state = (prior_folder, prior_state) if mode == 'normal' else (folder, state)
        values = {}
        for role in ['current', 'candidate']:
            name = role+'-'+mode
            value = read(data_folder/name/'result.json')
            source_run = next(r for r in owner_state['runs'] if r['name'] == name)
            assert source_run['complete'] and source_run['code'] == 0
            owner = source_run['processes']['worker']
            assert value['passed'] and value['role'] == role and value['mode'] == mode and value['pid'] == owner['pid']
            assert value['runtime'] == '10.0.8' and value['fma'] == (mode != 'scalar') and value['avx512'] == (mode == 'normal')
            assert value['core_sha256'] == built['products'][role]['sha256'] and value['consumer_sha256'] == built['consumer']['sha256']
            assert not value['performance_admitted']
            assert len(value['public_cases']) == 45 and len({r['name'] for r in value['public_cases']}) == 45
            for row in value['public_cases']:
                assert row['exact'] and row['owned_outputs'] and row['immutable_inputs']
                assert all(type(row[k]) is int and row[k] >= 0 for k in ['copies', 'scratches', 'allocated', 'checked_values'])
                assert pin(data_folder/name/('public-'+row['name']+'.f32')) == dict(bytes=4*row['checked_values'], sha256=row['output_sha256'])
            raw = value['raw_cases']
            assert len(raw) == (81 if role == 'candidate' and mode != 'scalar' else 0)
            assert len({r['name'] for r in raw}) == len(raw)
            for row in raw:
                assert row['exact'] and row['guards'] and row['immutable_inputs'] and row['allocated_bytes_for_eight_calls'] == 0
                assert row['checked_values'] == row['k']
                assert pin(data_folder/name/('raw-'+row['name']+'.f32')) == dict(bytes=4*row['k'], sha256=row['output_sha256'])
            values[role] = value
        before, after = values['current']['public_cases'], values['candidate']['public_cases']
        assert [r['name'] for r in before] == [r['name'] for r in after]
        for current, candidate in zip(before, after, strict=True):
            assert current['shape'] == candidate['shape'] and current['output_sha256'] == candidate['output_sha256']
            assert (data_folder/('current-'+mode)/('public-'+current['name']+'.f32')).read_bytes() == (data_folder/('candidate-'+mode)/('public-'+candidate['name']+'.f32')).read_bytes()
        reports[mode] = values
    analysis = dict(passed=True, product_changed_in_isolation=True, root_product_changed=False,
        performance_admitted=False, products=built['products'], consumer=built['consumer'],
        compiled=compiled, contracts=reports, resources=resources,
        prior_build=pin(FIRST/'failed.json'), prior_inventory_and_normal=pin(SECOND/'failed.json'),
        scope='One prepared single-row reader; complete exact public and guarded raw contracts. No timing score or model/application admission.')
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in folder.rglob('*') if p.is_file()}
    for name in ['analysis.json', 'prepared.json', 'payload.tar.gz', 'payload.json', 'staged.json', 'stage-started.json',
        'extracted.json', 'deployment.json', 'collection-transfer.json', 'results.tar.gz']:
        files[name] = pin(BASE/name)
    save(BASE/'closed.json', dict(passed=True, remote_terminal=True, performance_admitted=False, analysis=pin(BASE/'analysis.json'), files=files))
    print(json.dumps(dict(passed=True, closed=pin(BASE/'closed.json'), products=built['products'],
        public_cases=270, guarded_raw_cases=162, resources=sum(r['samples'] for r in resources), performance_admitted=False)))


if __name__ == '__main__': main()

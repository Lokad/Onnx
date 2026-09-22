"""Reconcile all raw/model qualification, resource evidence and fixed speed gates."""
import json
from run import BASE, prepared
from protocol import check_sample, pin, read, save
from score import ORDER, validate_and_score

QUALIFICATION = ['raw-256', 'raw-512', 'model-256', 'model-512']


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    tests = read(BASE/'selftest.json'); assert tests['passed'] and tests['tests'] == 12
    payload = read(BASE/'payload/payload.json'); collected = BASE/'collected'; receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    state = read(collected/'identity.json'); assert state['complete'] and state['code'] == 0 and state['ended']-state['started'] < 4*3600
    assert state['supervisor'] == read(BASE/'deployment.json') and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == QUALIFICATION+ORDER
    assert receipt['identities'] == [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    model = read(BASE/'payload/windows-256.json'); raw = read(BASE/'payload/windows-raw.json')
    resources = []; reports = {}; qualification = {}
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
        assert result['runtime'] == '10.0.8' and not result['flags']
        assert result['core'] == payload['core']['sha256'] and result['executable'] == payload['consumer']['sha256'] and result['pid'] == row['child']['pid']
        if row['name'] in QUALIFICATION:
            mode, width = row['name'].split('-'); expected = raw if mode == 'raw' else model
            assert result['avx512'] and result['lanes'] == (8 if width == '256' else 16)
            assert result['observations'] == expected['observations'] and result['values'] == expected['values']
            if mode == 'raw':
                assert result['cases'] == 2648 and result['geometries'] == 312 and result['layout_cases'] == 331
                assert result['finite_kernel_cases'] == 2496 and result['nonfinite_fallback_cases'] == 152
                assert result['rejected'] == 10 and result['owned_outputs'] and result['supplemental'] == expected['supplemental']
                assert result['failed_cases'] == result['scalar_differences'] == result['production_differences'] == 0
            else:
                assert result['cases'] == 108 and result['values'] == 119823360 and result['read_only_operands']
                assert result['failed'] == result['differences'] == result['native_failures'] == 0 and result['maximum'] <= 1e-4
            qualification[row['name']] = dict(cases=result['cases'], values=result['values'], lanes=result['lanes'], passed=True)
        else:
            assert result['lanes'] == 16 and result['read_only_operands']
            journal = [json.loads(s) for s in (collected/row['name']/'journal.jsonl').read_text().splitlines()]
            assert journal == result['preparation']+result['observations']; reports[row['name']] = result
        resources.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss'], seconds=row['seconds']))
    analysis = validate_and_score(reports, read(BASE/'payload/fixtures/result.json'), model)
    assert analysis['iteration_manifest'] == read(BASE/'payload/fixtures/iterations.json')
    analysis.update(complete=True, numerical_and_resource_checks_pass=True, qualification=qualification, resources=resources,
        samples=sum(r['samples'] for r in resources), peak_rss=max(r['peak_rss'] for r in resources),
        no_application_speed_measurement=True, no_ort_speed_measurement=True)
    save(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json', dict(passed=True, admitted=analysis['admitted'], files=files, local_inputs=spec['files'],
        remote_terminal=receipt['identities'], analysis=pin(BASE/'analysis.json'), scope='Complete evidence; only admitted permits product qualification.'))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), admitted=analysis['admitted'], aggregate=analysis['rows'][0],
        failed_controls=[r for r in analysis['controls'] if not r['passed']], failed_gates=[r for r in analysis['gates'] if not r['passed']],
        samples=analysis['samples'], peak_rss=analysis['peak_rss'])))


if __name__ == '__main__': main()

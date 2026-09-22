"""Recompute all qualification gates and matched raw-clock timings after collection."""
from fractions import Fraction
import json
from pathlib import Path
import sys
from candidate_protocol import LIMITS, ROLES, TIMING_ROLES, REQUIRED_TESTS, gate, pin, read, write, verified_files, check_sample, test_results
from transport import BASE, PREPARED, checked_local


def timing_table(results, manifest):
    assert len(results) == len(TIMING_ROLES)
    table = []
    for case in manifest['cases']:
        means = {}; row = dict(name=case['name'], audio_seconds=case['samples']/16000)
        for role in (*ROLES, 'ort'):
            processes = []; values = []
            for index, (assigned, result) in enumerate(zip(TIMING_ROLES, results, strict=True)):
                if role != assigned:
                    continue
                ticks = [Fraction(r['end_ticks']-r['start_ticks'], r['frequency']) for r in result['records']
                         if r['name'] == case['name'] and r['phase'] == 'measured']
                assert len(ticks) == 3 and all(t > 0 for t in ticks)
                values.extend(ticks)
                processes.append(dict(index=index, seconds=[float(t) for t in ticks], mean=float(sum(ticks)/3)))
            assert len(values) == 6
            means[role] = sum(values)/6
            row[role] = dict(seconds=float(means[role]), rtf=float(means[role]/Fraction(case['samples'], 16000)),
                             minimum=float(min(values)), maximum=float(max(values)), processes=processes)
        row['ratios_to_ort'] = {role: float(means[role]/means['ort']) for role in ROLES}
        row['ratios_to_production'] = {role: float(means[role]/means['production']) for role in ROLES[1:]}
        table.append(row)
    return table


def main():
    prepared, bundle, execution = checked_local()
    root = BASE/'collected'; payload = PREPARED/'payload'; campaign = root/'campaign'
    receipt = read(root/'collection.json'); transfer = read(BASE/'collection-transfer.json')
    assert transfer['archive'] == pin(BASE/'results.tar.gz') and transfer['receipt'] == pin(root/'collection.json')
    assert receipt['terminal'] is True
    verified_files(root, receipt['files']); verified_files(payload, read(payload/'payload.json')['files'])
    assert receipt['payload'] == prepared['payload'] and receipt['execution'] == bundle['execution']
    state_path = campaign/'identity.json'
    state = read(state_path) if state_path.exists() else None
    if receipt['input_error'] is not None or state is None or state['code'] != 0:
        failure = dict(audit_completed=True, campaign_passed=False, timing_verdict=False,
                       collection=pin(root/'collection.json'), input_error=receipt['input_error'],
                       supervisor_error=None if state is None else state.get('error'),
                       retained_runs=0 if state is None else len(state['runs']))
        write(BASE/'failure-audit.json', failure); print(json.dumps(failure)); return 1
    assert state['complete'] is True and state['limits'] == LIMITS
    assert state['execution'] == bundle['execution'] and state['payload'] == prepared['payload']
    expected = ['sdk-version']+[n+s for n in ('backend', 'tensors', 'cli', 'il-bridge') for s in ('-restore', '-build')]
    expected += ['il-bridge', 'backend-tests', 'tensors-tests']
    expected += ['caller-normal', 'caller-disabled'] + [r+'-'+f for r in ROLES for f in ['pyannote', 'parakeet']]
    expected += ['native-conformance', 'meetings-inputs', 'meetings-run']+[f'timing-{i:02}-{r}' for i, r in enumerate(TIMING_ROLES)]
    assert [r['name'] for r in state['runs']] == expected
    assert state['seconds'] < LIMITS['campaign_seconds']
    samples = 0
    for run in state['runs']:
        assert run['complete'] is True and run['code'] == 0
        assert run['preflight_available'] >= LIMITS['preflight_available'] and run['preflight_tmpfs_free'] >= LIMITS['preflight_tmpfs_free']
        resource = [json.loads(s) for s in (campaign/run['name']/'samples.jsonl').read_text().splitlines()]
        assert len(resource) == run['samples'] > 0
        assert max(sum(m['rss'] for m in s['members']) for s in resource) == run['peak_rss']
        for sample in resource:
            check_sample(sample)
            for member in sample['members']:
                assert run['members'][str(member['pid'])] == member['birth']
        # Setup/build commands may block logging longer than inference; keep all
        # gaps as evidence, and require regular sampling for graph/timing work.
        if run['name'].startswith(('production-', 'portable-', 'rows-', 'native-', 'timing-', 'meetings-')):
            gaps = [resource[0]['seconds']]+[b['seconds']-a['seconds'] for a, b in zip(resource, resource[1:])]
            gaps += [run['seconds']-resource[-1]['seconds']]
            assert all(0 <= g < 10 for g in gaps), (run['name'], gaps)
        if 'terminal_transition' in run:
            from terminal_snapshot import validate
            event = read(campaign/run['name']/'terminal-transition.json')
            validate(event)
            assert event == run['terminal_transition']
            assert event['identities'] == run['members'] and event['code'] == run['code']
            assert event['seconds'] <= run['seconds']
        samples += len(resource)
    built = read(campaign/'built-files.json'); verified_files(root, built)
    for name in ('Lokad.Onnx.CLI.dll', 'Lokad.Onnx.CLI.deps.json', 'Lokad.Onnx.CLI.runtimeconfig.json'):
        assert 'source/src/Lokad.Onnx.CLI/bin/Release/net10.0/'+name in built
    bridge = read(campaign/'il-bridge.json'); assert bridge['passed'] is True
    for observation in bridge['observations']:
        name = observation['assembly']; assert observation['equal'] is True
        assert observation['before_sha256'] == pin(payload/'runtimes/portable'/name)['sha256']
        assert observation['after_sha256'] == pin(root/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)['sha256']
    operator_gate = read(campaign/'operator-gate.json')
    assert operator_gate['passed'] is True and operator_gate['il'] == pin(campaign/'il-bridge.json')
    for suite in ('backend', 'tensors'):
        result = test_results(campaign/'test-results'/(suite+'.trx'), 3313 if suite == 'backend' else 343,
                              REQUIRED_TESTS if suite == 'backend' else ())
        assert result == operator_gate['suites'][suite]
    # Use the exact bundled auditor and original native protocol.
    sys.path.insert(0, str(BASE/'execution')); sys.path.insert(0, str(payload/'runtime'))
    from qualify_outputs import pyannote, parakeet
    from protocol import validate_records
    from fresh_qualification import qualification
    reports = qualification(payload, campaign)
    gate(reports)
    gate_receipt = read(campaign/'qualification-gate.json')
    assert gate_receipt['passed'] is True and gate_receipt['reports'] == reports
    assert gate_receipt['operator_gate'] == pin(campaign/'operator-gate.json')
    results = []
    for label, role, mode in [('native-conformance', 'ort', 'conformance')]+[
            (f'timing-{i:02}-{r}', r, 'timing') for i, r in enumerate(TIMING_ROLES)]:
        manifest_path = payload/'manifests'/('production-pyannote.json' if role == 'ort' else role+'-pyannote.json')
        manifest = read(manifest_path); folder = campaign/(label+'-output'); result = read(folder/'result.json')
        validate_records(result, manifest, mode)
        assert result['manifest_sha256'] == pin(manifest_path)['sha256']
        assert result['engine'] == ('ort' if role == 'ort' else 'managed')
        for index, row in enumerate(result['records']):
            assert row == read(folder/f'{index:03}.json')
        if role == 'ort':
            spec = read(payload/'payload.json')
            assert result['python_binary'] == spec['interpreter'] and result['runner_sha256'] == pin(payload/'runtime/native.py')['sha256']
            assert result['versions'] == manifest['native_versions'] and result['native_binaries'] == manifest['native_binaries']
            for name, wanted in result['numeric_libraries'].items():
                assert spec['external'][name] == wanted
        else:
            assert result['runtime'] == '.NET 10.0.8' and result['processor_count'] == 1
            assert result['runner_sha256'] == pin(payload/'runtimes'/role/'AudioBenchmark.dll')['sha256']
            assert all(result[k] == manifest[k] for k in ('core_sha256', 'data_sha256'))
        if mode == 'timing':
            results.append(result)
        else:
            assert gate_receipt['native'] == pin(folder/'result.json')
    table = timing_table(results, read(payload/'manifests/production-pyannote.json'))
    from meetings_audit import audit_meetings
    meetings = audit_meetings(payload, campaign)
    assert meetings == read(campaign/'meetings-audit.json') and meetings['passed']
    from admission import evaluate
    performance = evaluate(table)

    analysis = dict(passed=True, role_labels=read(payload/'payload.json')['role_labels'], meetings=meetings, performance=performance, fresh_arrays_pyannote=36, fresh_arrays_parakeet=1568, fresh_pyannote_public_calls=32, fresh_pyannote_public_qualification_calls=4,
                    parakeet_public_qualification_cases=21,
                    timing_calls=96, measured=72, warmup=24, table=table, reports=reports, resource_samples=samples,
                    maximum_centroid_error=max(r['maximum_centroid_error'] for result in results for r in result['records']),
                    peak_rss=max(r['peak_rss'] for r in state['runs']), operator_tests=operator_gate['suites'],
                    accounting={r['name']: r['accounting'] for r in state['runs']},
                    limitation='Matched AMD comparison with prospective repeatability and speed admission; no calibrated parity claim')
    write(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and 'controller' not in p.relative_to(BASE).parts}
    write(BASE/'closed.json', dict(passed=True, files=files, collection=pin(root/'collection.json'), analysis=pin(BASE/'analysis.json')))
    print(json.dumps({k: v for k, v in analysis.items() if k not in ('reports', 'accounting', 'operator_tests')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

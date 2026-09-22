"""Recompute all qualification gates and matched raw-clock timings after collection."""
from fractions import Fraction
import json
from pathlib import Path
import sys
from candidate_protocol import LIMITS, ROLES, TIMING_ROLES, REQUIRED_TESTS, gate, pin, read, write, verified_files, check_sample, test_results
from transport import BASE, PREPARED, checked_local


def timing_table(results, manifest):
    assert len(results) == len(TIMING_ROLES)
    def rational(value): return dict(numerator=value.numerator, denominator=value.denominator)
    table = []
    cases = manifest['cases']; assert len({c['name'] for c in cases}) == len(cases)
    for case in [*cases, dict(name='complete-corpus', samples=sum(c['samples'] for c in cases), is_corpus=True)]:
        full = case.get('is_corpus', False)
        names = {c['name'] for c in cases} if full else {case['name']}
        means = {}; row = dict(name=case['name'], audio_seconds=case['samples']/16000, is_corpus=full)
        for role in (*ROLES, 'ort'):
            processes = []; values = []
            for index, (assigned, result) in enumerate(zip(TIMING_ROLES, results, strict=True)):
                if role != assigned: continue
                by_case = {name: [Fraction(r['end_ticks']-r['start_ticks'], r['frequency']) for r in result['records']
                    if r['name'] == name and r['phase'] == 'measured'] for name in names}
                assert all(len(t) == 3 and all(v > 0 for v in t) for t in by_case.values())
                ticks = [sum(t[i] for t in by_case.values()) for i in range(3)]
                mean = sum(ticks)/3; values.extend(ticks)
                processes.append(dict(index=index, seconds=[float(t) for t in ticks], mean=float(mean), exact_mean=rational(mean)))
            assert len(values) == 6
            means[role] = sum(values)/6
            row[role] = dict(seconds=float(means[role]), exact_mean=rational(means[role]),
                rtf=float(means[role]/Fraction(case['samples'], 16000)),
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
    expected += ['caller-normal', 'probe-normal', 'caller-disabled', 'probe-disabled'] + [r+'-'+f for r in ROLES for f in ['pyannote', 'parakeet']]
    expected += ['native-conformance']+[r+'-parakeet-public' for r in (*ROLES, 'ort')]+['meetings-inputs', 'meetings-run']+[f'timing-{i:02}-{r}' for i, r in enumerate(TIMING_ROLES)]
    assert [r['name'] for r in state['runs']] == expected
    assert state['seconds'] < LIMITS['campaign_seconds']
    samples = 0
    for run in state['runs']:
        assert run['complete'] is True and run['code'] == 0
        minimum = LIMITS['public_preflight_available'] if run['name'].endswith('-parakeet-public') or run['name'].startswith('timing-') else LIMITS['preflight_available']
        assert run['preflight_minimum'] == minimum
        assert run['preflight_available'] >= minimum and run['preflight_tmpfs_free'] >= LIMITS['preflight_tmpfs_free']
        preflight = read(campaign/(run['name']+'-preflight.json'))
        assert preflight == dict(minimum=minimum, observations=run['preflight_observations'])
        assert preflight['observations'] and preflight['observations'][-1]['available'] >= minimum
        assert all(0 <= r['seconds'] < 900 and r['tmpfs_free'] >= LIMITS['preflight_tmpfs_free'] for r in preflight['observations'])
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
    assert [(r['assembly'], r['methods'], r['equal']) for r in bridge['observations']] == [('Lokad.Onnx.dll', 3114, True), ('Lokad.Onnx.Data.dll', 697, True)]
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
    from applications import inspect as inspect_application, conformance
    public = conformance(payload, campaign)
    assert public == read(campaign/'public-conformance.json')
    assert gate_receipt['public'] == pin(campaign/'public-conformance.json')
    assert gate_receipt['native'] == public['pyannote_native']
    results = [inspect_application(payload, campaign, f'timing-{i:02}-{r}', r, 'parakeet', 'timing')
               for i, r in enumerate(TIMING_ROLES)]
    assert sum(len(r['records']) for r in results) == 480
    assert sum(v['phase'] == 'measured' for r in results for v in r['records']) == 360
    table = timing_table(results, read(payload/'manifests/production-parakeet.json'))
    from meetings_audit import audit_meetings
    meetings = audit_meetings(payload, campaign)
    assert meetings == read(campaign/'meetings-audit.json') and meetings['passed']
    from admission import evaluate
    performance = evaluate(table)

    analysis = dict(passed=True, role_labels=read(payload/'payload.json')['role_labels'], meetings=meetings, performance=performance, fresh_arrays_pyannote=36, fresh_arrays_parakeet=1568, fresh_pyannote_public_calls=32, fresh_pyannote_public_qualification_calls=4,
                    parakeet_native_trajectory_reference_cases=21, fresh_parakeet_public_conformance=public,
                    timing_calls=480, measured=360, warmup=120, table=table, reports=reports, resource_samples=samples,
                    maximum_centroid_error=max(r['maximum_centroid_error'] for result in results for r in result['records']),
                    peak_rss=max(r['peak_rss'] for r in state['runs']), operator_tests=operator_gate['suites'],
                    accounting={r['name']: r['accounting'] for r in state['runs']},
                    limitation='Matched AMD comparison with prospective repeatability and speed admission; no calibrated parity claim')
    write(BASE/'analysis.json', analysis)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file() and 'controller' not in p.relative_to(BASE).parts}
    write(BASE/'closed.json', dict(passed=True, files=files, collection=pin(root/'collection.json'), analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(passed=True, performance_admitted=performance['admitted'], corpus=table[-1], resource_samples=samples, closure=pin(BASE/'closed.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

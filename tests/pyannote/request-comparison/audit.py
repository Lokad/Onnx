"""Audit every original result and resource sample, retaining fixed controls."""
import json
import statistics
from common import *


def main():
    assert not (BASE / 'closed.json').exists() and not (BASE / 'analysis.json').exists()
    prepared = read(BASE / 'prepared.json')
    verify_prepared(prepared)
    state = read(BASE / 'processes.json')
    assert state['complete'] and state['code'] == 0
    terminal(state['supervisor'])
    assert [r['name'] for r in state['runs']] == [f'{i}-{role}' for i, role in enumerate(prepared['jobs'])]
    manifest = manifest_with_raw_hashes()
    auditor = public_auditor()
    expected = {r['name']: r['result'] for r in read(QUALIFIED / 'dialogue-output/result.json')['records']}
    identities = [state['supervisor']]
    workers, observations = [], []
    limits = prepared['limits']
    for index, (run, role) in enumerate(zip(state['runs'], prepared['jobs'], strict=True)):
        assert run['complete'] and run['code'] == 0 and run['application_passed'] and run['samples'] > 0
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        assert run['preflight'] == run['preflight_observations'][-1]
        assert run['preflight']['available'] >= limits['preflight_gib'] * 1024**3
        assert all(s['seconds'] < 900 and s['disk'] >= 20 * 1024**3 for s in run['preflight_observations'])
        folder = BASE / run['name']
        result = read(folder / 'output/result.json')
        auditor.validate_worker(result, manifest, 'timing')
        assert result['manifest_sha256'] == pin(INPUT)['sha256']
        if role == 'ort':
            assert result['engine'] == 'ort' and result['onnxruntime'] == '1.29.0' and result['numpy'] == '2.2.4'
            assert result['native_binaries'] == prepared['native_binaries'] and result['native_settings'] == prepared['native_settings']
            assert result['runner_sha256'] == pin(NATIVE)['sha256'] and result['adapter_sha256'] == pin(NATIVE.with_name('native_adapters.py'))['sha256']
            assert result['python_binary_sha256'] == prepared['external_files'][sys.executable]['sha256']
            assert result['flags'] == {k: '1' for k in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}
        else:
            assert result['engine'] == 'managed' and result['runtime'] == '.NET 10.0.12' and result['processor_count'] == 1 and not result['flags']
            for key, name in [('core_sha256', 'Lokad.Onnx.dll'), ('data_sha256', 'Lokad.Onnx.Data.dll'), ('runner_sha256', 'AudioBenchmark.dll')]:
                assert result[key] == prepared['roles'][role][name]['sha256']
        assert [p.name for p in sorted((folder / 'output').glob('[0-9][0-9][0-9].json'))] == [f'{i:03}.json' for i in range(16)]
        for ordinal, row in enumerate(result['records']):
            assert row == read(folder / 'output' / f'{ordinal:03}.json')
            if role != 'ort':
                assert row['result'] == expected[row['name']] and row['allocated_bytes'] >= 0
                assert all(b >= a for a, b in zip(row['gc_before'], row['gc_after'], strict=True))
            observations.append(dict(index=index, role=role, **{k: v for k, v in row.items() if k != 'result'}))
        samples = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] and max(s['rss'] for s in samples) == run['peak_rss']
        assert samples[-1]['seconds'] <= run['seconds'] < limits['seconds']
        for sample in samples:
            assert sample['seconds'] < limits['seconds'] and sample['rss'] < limits['rss_gib'] * 1024**3
            assert sample['available'] >= 1024**3 and sample['disk'] >= 20 * 1024**3 and sample['output_bytes'] <= 1024**3
            assert len(sample['members']) <= 1 and sample['rss'] == sum(p['rss'] for p in sample['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in sample['members'])
        gaps = [samples[0]['seconds']] + [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])] + [run['seconds'] - samples[-1]['seconds']]
        assert all(0 <= value < 10 for value in gaps)
        workers.append(dict(index=index, role=role, samples=run['samples'], peak_rss=run['peak_rss'],
            means={c['name']: statistics.fmean(r['seconds'] for r in result['records'] if r['name'] == c['name'] and r['phase'] == 'measured') for c in manifest['cases']}))
    assert len(observations) == 96 and sum(r['phase'] == 'measured' for r in observations) == 72
    controls, means = {}, {}
    for role in ['predecessor', 'candidate', 'ort']:
        rows = [w for w in workers if w['role'] == role]
        assert len(rows) == 2
        ratios = {c['name']: max(w['means'][c['name']] for w in rows) / min(w['means'][c['name']] for w in rows) for c in manifest['cases']}
        controls[role] = dict(fixture_max_min=ratios, passed=ratios['dialogue-30s'] <= prepared['controls']['full_request_max_ratio']
            and max(ratios.values()) <= prepared['controls']['fixture_max_ratio'])
        means[role] = {c['name']: statistics.fmean(w['means'][c['name']] for w in rows) for c in manifest['cases']}
    valid = all(c['passed'] for c in controls.values())
    table = []
    for case in manifest['cases']:
        name = case['name']
        row = dict(name=name, audio_seconds=case['samples'] / 16000,
            seconds={role: means[role][name] for role in means}, candidate_to_predecessor=means['candidate'][name] / means['predecessor'][name],
            candidate_to_ort=means['candidate'][name] / means['ort'][name], allocations={})
        for role in ['predecessor', 'candidate']:
            selected = [r for r in observations if r['role'] == role and r['name'] == name and r['phase'] == 'measured']
            assert len(selected) == 6
            row['allocations'][role] = statistics.fmean(r['allocated_bytes'] for r in selected)
        row['allocation_ratio'] = row['allocations']['candidate'] / row['allocations']['predecessor']
        table.append(row)
    admitted = valid and table[0]['candidate_to_predecessor'] <= prepared['admission']['full_request_ratio'] and max(r['candidate_to_predecessor'] for r in table) <= prepared['admission']['maximum_fixture_ratio']
    analysis = dict(passed=True, attribution_valid=valid, qualifies_for_later_amd=admitted, calls=96, warmup_calls=24, measured_calls=72,
        controls=controls, table=table, workers=workers, observations=observations,
        resource_samples=sum(r['samples'] for r in state['runs']), peak_rss=max(r['peak_rss'] for r in state['runs']),
        scope=prepared['scope'])
    save(BASE / 'analysis.json', analysis)
    files = dict(prepared['files'])
    for p in [*BASE.rglob('*'), *TOOLS.iterdir()]:
        if p.is_file():
            files[p.relative_to(ROOT).as_posix()] = pin(p)
    save(BASE / 'closed.json', dict(passed=True, attribution_valid=valid, files=files, external_files=prepared['external_files'],
        terminal_identities=identities, analysis=pin(BASE / 'analysis.json')))
    print(json.dumps({k: analysis[k] for k in ['passed', 'attribution_valid', 'qualifies_for_later_amd', 'calls', 'resource_samples', 'peak_rss', 'controls', 'table']}))
    print(json.dumps(dict(closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()

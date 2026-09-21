"""Audit complete public results and independently account every exported stack interval."""
import hashlib
import statistics
from common import *
from stacks_v2 import inspect, cross_export
import numpy as np

MARKERS = {
    'dialogue-30s': '!SampledRequests.Full(',
    'dialogue-0-10s': '!SampledRequests.FirstCrop(',
    'dialogue-10-20s': '!SampledRequests.SecondCrop(',
    'dialogue-20-30s': '!SampledRequests.ThirdCrop(',
}


def main():
    assert not (BASE / 'model-closed.json').exists()
    spec = read(BASE / 'model-prepared.json')
    verify_spec(spec)
    identities, resources = [], []
    for filename, expected in [
        ('model-preparation.json', ['consumer-restore', 'consumer-build']),
        ('model-processes.json', spec['jobs'] + [name+'-'+format for name in spec['jobs'][1:] for format in ['speedscope', 'chromium']])]:
        state = read(BASE / filename)
        assert state['complete'] and state['code'] == 0 and [r['name'] for r in state['runs']] == expected
        identities.append(state['supervisor'])
        for run in state['runs']:
            paired = run['name'] in spec['jobs']
            assert run['complete'] and run['code'] == 0 and run['preflight']['available'] >= (10 if paired else 8)*1024**3
            assert run['preflight'] == run['preflight_observations'][-1]
            assert all(s['seconds'] < 900 and s['disk'] >= 20*1024**3 for s in run['preflight_observations'])
            identities.extend(run['processes'].values() if paired else [dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items()])
            samples = [json.loads(s) for s in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
            assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
            assert samples[-1]['seconds'] <= run['seconds'] < 900
            for sample in samples:
                assert sample['rss'] < 8*1024**3 and sample['available'] >= 1024**3 and sample['disk'] >= 20*1024**3 and sample['output_bytes'] <= 1024**3
                assert sample['rss'] == sum(p['rss'] for p in sample['members'])
                for p in sample['members']:
                    member = run['members'][str(p['pid'])]
                    if paired:
                        assert member['birth'] == p['birth'] and member['role'] == p['role']
                        assert member['affinity'] == p['affinity'] == ([2] if p['role']=='target' else [0])
                    else:
                        assert member == p['birth'] and p['affinity'] == [2]
            if paired:
                sampled = run['name'] != 'control'
                assert run['exit_codes'] == ({'target':0,'collector':0} if sampled else {'target':0})
                assert run['ready'] == read(BASE / run['name'] / 'ready.json') and run['ready']['warmup_records'] == 4
                if sampled:
                    assert run['enabled'] == read(BASE / run['name'] / 'collector-enabled.json') and run['enabled']['enabled']
            resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss'], seconds=run['seconds']))
    for identity in identities:
        terminal(identity)
    module_spec = importlib.util.spec_from_file_location('original_public_audit', ROOT / 'tests/audio/comparison/audit.py')
    auditor = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(auditor)
    manifest = read(INPUT)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    original = {r['name']:r['result'] for r in read(QUALIFIED / 'dialogue-output/result.json')['records']}
    results, diagnostics = {}, []
    for name in spec['jobs']:
        output = BASE / name
        result = read(output / 'result.json')
        results[name] = result
        auditor.validate_worker(result, manifest, 'timing')
        assert result['passed'] and result['sampled'] == (name!='control') and result['runtime'] == '.NET 10.0.12'
        assert result['core_sha256'] == CORE and result['data_sha256'] == DATA and result['runner_sha256'] == spec['consumer']['sha256']
        assert result['manifest_sha256'] == pin(INPUT)['sha256'] and not result['flags'] and result['processor_count'] == 1
        assert {p.name for p in output.glob('[0-9][0-9][0-9].json')} == {f'{i:03}.json' for i in range(16)}
        for index, row in enumerate(result['records']):
            assert row == read(output / f'{index:03}.json') and row['result'] == original[row['name']]
            assert row['cpu_user_ticks'] >= 0 and row['cpu_system_ticks'] >= 0 and row['cpu_frequency'] == 10000000
            assert (row['cpu_user_ticks']+row['cpu_system_ticks'])/row['cpu_frequency'] <= row['seconds']+.1
            assert row['thread_id'] == read(output / 'ready.json')['thread_id']
        if name == 'control':
            assert not (output / 'capture.nettrace').exists()
            continue
        speedscope, chromium = read(output / 'speedscope.speedscope.json'), read(output / 'chromium.chromium.json')
        exported = cross_export(speedscope, chromium)
        parsed = inspect(speedscope, MARKERS)
        assert not any('SampledRequests.Warmup(' in f['name'] for f in speedscope['shared']['frames'])
        target = read(output / 'ready.json')
        process_frame = [f['name'] for f in speedscope['shared']['frames'] if f['name'].startswith('Process64 dotnet (')]
        assert len(process_frame) == 1 and f"({target['pid']})" in process_frame[0]
        coverage = []
        for case, intervals in parsed['intervals'].items():
            assert all(i['thread'] == f"Thread ({target['thread_id']})" for i in intervals)
            selected = [r for r in result['records'] if r['name']==case and r['phase']=='measured']
            assert len(selected) == 3
            wall = sum(r['seconds'] for r in selected)
            sampled_seconds = parsed['selected_seconds'][case]
            # Three complete repetitions are required; losing one full call
            # cannot be admitted as a complete-workload diagnostic.
            tolerance = min(.15, min(r['seconds'] for r in selected)/wall/2)
            assert abs(sampled_seconds/wall-1) < tolerance, (name, case, sampled_seconds, wall, tolerance)
            coverage.append(dict(name=case, calls=3, wall_seconds=wall, sampled_thread_seconds=sampled_seconds,
                process_cpu_seconds=sum(r['cpu_user_ticks']+r['cpu_system_ticks'] for r in selected)/10000000,
                sampled_to_wall=sampled_seconds/wall, marker_intervals=len(intervals)))
        diagnostics.append(dict(name=name, exports=exported, coverage=coverage, **parsed))
    observations = []
    for case in MARKERS:
        roles = {}
        for name, result in results.items():
            rows = [r for r in result['records'] if r['name']==case and r['phase']=='measured']
            roles[name] = dict(wall_mean=statistics.fmean(r['seconds'] for r in rows),
                process_cpu_mean=statistics.fmean((r['cpu_user_ticks']+r['cpu_system_ticks'])/10000000 for r in rows),
                allocated_mean=statistics.fmean(r['allocated_bytes'] for r in rows))
        observations.append(dict(name=case, roles=roles,
            diagnostic_to_control={name: roles[name]['wall_mean']/roles['control']['wall_mean'] for name in spec['jobs'][1:]}))
    analysis = dict(passed=True, calls=48, measured_calls=36, exact_qualified_results=True, diagnostics=diagnostics,
        observations=observations, resources=resources, resource_samples=sum(r['samples'] for r in resources),
        identities=identities, scope='Sampled managed thread-time attribution, with process CPU separately measured; no speedup, ORT ratio or AMD admission.')
    save(BASE / 'model-analysis.json', analysis)
    files = dict(spec['files'])
    for folder in [BASE, TOOLS]:
        for path in folder.rglob('*'):
            if path.is_file() and not {'obj','packages'}.intersection(path.relative_to(folder).parts):
                files[rel(path)] = pin(path)
    save(BASE / 'model-closed.json', dict(passed=True, files=files, external_files=spec['external_files'], identities=identities,
        analysis=pin(BASE / 'model-analysis.json')))
    print(dict(passed=True, calls=48, resource_samples=analysis['resource_samples'], closed=pin(BASE / 'model-closed.json')))
    for diagnostic in diagnostics:
        print(diagnostic['name'], diagnostic['coverage'])
        print('Full-request top leaves', [r for r in diagnostic['exclusive'] if r['marker']=='dialogue-30s'][:12])


if __name__ == '__main__':
    main()

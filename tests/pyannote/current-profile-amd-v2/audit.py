"""Reconcile complete AMD public requests, resources and both stack exports."""
import hashlib
import statistics
import sys
from common import *
import numpy as np

MARKERS = {'dialogue-30s':'!SampledRequests.Full(', 'dialogue-0-10s':'!SampledRequests.FirstCrop(',
    'dialogue-10-20s':'!SampledRequests.SecondCrop(', 'dialogue-20-30s':'!SampledRequests.ThirdCrop('}


def main():
    spec = prepared(); assert not (BASE / 'closed.json').exists()
    payload = BASE / 'payload'; collected = BASE / 'collected'
    transfer = read(BASE / 'collection-transfer.json'); receipt = read(collected / 'collection.json')
    assert transfer['terminal'] and transfer['code'] == 0 and transfer['input_error'] is None
    assert transfer['receipt'] == pin(collected / 'collection.json') and transfer['archive'] == pin(BASE / 'results.tar.gz')
    for name, wanted in receipt['files'].items(): assert pin(collected / name) == wanted, name
    state = read(collected / 'identity.json')
    assert state['complete'] and state['code'] == 0 and state['boot_time'] == 1789634288.0
    assert [r['name'] for r in state['runs']] == ['control','sampled-a','sampled-b']
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert receipt['payload'] == spec['payload']
    assert state['supervisor'] == {k: read(collected / 'deployment.json')[k] for k in ['pid','birth']}
    assert receipt['identities'] == [read(collected / 'deployment.json')] + [i for r in state['runs'] for i in r['processes'].values()]
    resources = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        assert run['preflight']['available'] >= 10*1024**3 and run['preflight']['disk'] >= 1024**3
        assert run['preflight'] == run['preflight_observations'][-1]
        assert all(r['seconds'] < 900 and r['disk'] >= 1024**3 for r in run['preflight_observations'])
        rows = [json.loads(s) for s in (collected / 'logs' / (run['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        assert rows[-1]['seconds'] <= run['seconds']
        for row in rows:
            assert row['seconds'] < 900 and row['rss'] < 8*1024**3 and row['available'] >= 1024**3
            assert row['disk'] >= 1024**3 and row['output_bytes'] <= 1024**3
            assert row['rss'] == sum(m['rss'] for m in row['members'])
            for member in row['members']:
                original = run['members'][str(member['pid'])]
                assert original['birth'] == member['birth'] and original['role'] == member['role']
                assert original['affinity'] == member['affinity'] == ([2] if member['role']=='target' else [0])
                assert member['threads'] and all(t['affinity'] == member['affinity'] for t in member['threads'])
        sampled = run['name'] != 'control'
        assert run['exit_codes'] == ({'target':0,'collector':0} if sampled else {'target':0})
        ready = read(collected / run['name'] / 'ready.json')
        assert ready == run['ready'] and ready['warmup_records'] == 4 and ready['runtime'] == '10.0.8' and ready['affinity'] == 4
        assert ready['pid'] == run['processes']['target']['pid']
        assert abs(ready['birth_milliseconds']/1000-run['processes']['target']['birth']) < 1.1
        assert not ready['flags']
        if sampled:
            assert run['enabled'] == read(collected / run['name'] / 'collector-enabled.json') and run['enabled']['enabled']
        resources.append(dict(name=run['name'], samples=len(rows), peak_rss=run['peak_rss'], seconds=run['seconds']))
    from prepare import BUILD
    from export_audit import inspect as inspect_exports
    build = read(BUILD/'analysis.json')
    assert build['passed'] and build['no_profile_capture'] and build['root_product_changed'] is False
    exports = inspect_exports()
    public = module('original_public_auditor', payload / 'tools/public_audit.py')
    sys.path.insert(0, str(payload / 'tools'))
    from selected_stacks import inspect, cross_export
    manifest = read(payload / 'manifest.json')
    for case in manifest['cases']:
        pcm = np.load(payload / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    previous = {r['name']:r['result'] for r in read(payload / 'prior-amd-result.json')['records']}
    results, diagnostics = {}, []
    for name in ['control','sampled-a','sampled-b']:
        output = collected / name; result = read(output / 'result.json'); results[name] = result
        public.validate_worker(result, manifest, 'timing')
        assert result['passed'] and result['sampled'] == (name != 'control') and result['runtime'] == '.NET 10.0.8'
        assert result['core_sha256'] == CORE and result['data_sha256'] == DATA
        assert result['runner_sha256'] == pin(payload / 'runtime/SampledAudio.dll')['sha256']
        assert result['manifest_sha256'] == pin(payload / 'manifest.json')['sha256']
        assert not result['flags'] and result['processor_count'] == 1
        assert {p.name for p in output.glob('[0-9][0-9][0-9].json')} == {f'{i:03}.json' for i in range(16)}
        for index, row in enumerate(result['records']):
            assert row == read(output / f'{index:03}.json') and row['result'] == previous[row['name']]
            assert row['cpu_user_ticks'] >= 0 and row['cpu_system_ticks'] >= 0 and row['cpu_frequency'] == 10000000
            assert (row['cpu_user_ticks']+row['cpu_system_ticks'])/row['cpu_frequency'] <= row['seconds']+.1
            assert row['thread_id'] == read(output / 'ready.json')['thread_id']
        if name == 'control':
            assert not (output / 'capture.nettrace').exists(); continue
        folder = BASE / 'exports' / name
        speedscope, chromium = read(folder / 'speedscope.speedscope.json'), read(folder / 'chromium.chromium.json')
        exported = cross_export(speedscope, chromium); parsed = inspect(speedscope, MARKERS)
        assert not any('SampledRequests.Warmup(' in f['name'] for f in speedscope['shared']['frames'])
        target = read(output / 'ready.json')
        process_frame = [f['name'] for f in speedscope['shared']['frames'] if f['name'].startswith('Process64 dotnet (')]
        assert len(process_frame) == 1 and f"({target['pid']})" in process_frame[0]
        coverage = []
        for case, intervals in parsed['intervals'].items():
            assert all(i['thread'] == f"Thread ({target['thread_id']})" for i in intervals)
            selected = [r for r in result['records'] if r['name']==case and r['phase']=='measured']; assert len(selected) == 3
            wall = sum(r['seconds'] for r in selected); sampled_seconds = parsed['selected_seconds'][case]
            tolerance = min(.15, min(r['seconds'] for r in selected)/wall/2)
            assert abs(sampled_seconds/wall-1) < tolerance, (name,case,sampled_seconds,wall,tolerance)
            coverage.append(dict(name=case,calls=3,wall_seconds=wall,sampled_thread_seconds=sampled_seconds,
                process_cpu_seconds=sum(r['cpu_user_ticks']+r['cpu_system_ticks'] for r in selected)/10000000,
                sampled_to_wall=sampled_seconds/wall,marker_intervals=len(intervals)))
        diagnostics.append(dict(name=name,exports=exported,coverage=coverage,**parsed))
    observations = []
    for case in MARKERS:
        roles = {}
        for name, result in results.items():
            rows = [r for r in result['records'] if r['name']==case and r['phase']=='measured']
            roles[name] = dict(wall_mean=statistics.fmean(r['seconds'] for r in rows),
                process_cpu_mean=statistics.fmean((r['cpu_user_ticks']+r['cpu_system_ticks'])/10000000 for r in rows),
                allocated_mean=statistics.fmean(r['allocated_bytes'] for r in rows))
        observations.append(dict(name=case,roles=roles,diagnostic_to_control={n:roles[n]['wall_mean']/roles['control']['wall_mean'] for n in ['sampled-a','sampled-b']}))
    inventory = read(BUILD / 'collected/inventory/instructions.json')['observations'][0]
    assert inventory['public_surface_equal'] and not inventory['removed'] and len(inventory['differences']) == 1
    assert not inventory['added'] and inventory['methods'] == 160 and inventory['unchanged_methods'] == 159
    assert inventory['differences'][0].startswith('Program::<Main>$::')
    analysis = dict(passed=True,calls=48,measured_calls=36,exact_prior_amd_results=True,diagnostics=diagnostics,
        observations=observations,resources=resources+build['resources']+exports['resources'],
        remote_identities=receipt['identities']+exports['identities']+read(BUILD/'collected/collection.json')['identities'],local_identities=[],
        consumer=dict(existing_methods=inventory['methods'],unchanged_methods=inventory['unchanged_methods'],added_imports=0,changed_hash_literals=2),
        scope='AMD sampled managed thread time; process CPU separately measured; no new ORT ratio or speed selection')
    save(BASE / 'analysis.json', analysis)
    files = dict(spec['files'])
    files.update({rel(p):pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)})
    save(BASE / 'closed.json', dict(passed=True,files=files,analysis=pin(BASE / 'analysis.json'),
        remote_identities=analysis['remote_identities'],local_identities=[]))
    print(json.dumps(dict(passed=True,calls=48,samples=sum(r['samples'] for r in analysis['resources']),closed=pin(BASE / 'closed.json'))))
    for diagnostic in diagnostics:
        print(diagnostic['name'],diagnostic['coverage'])
        print('Top full-request leaves', [r for r in diagnostic['exclusive'] if r['marker']=='dialogue-30s'][:12])


if __name__ == '__main__': main()

"""Independently audit the complete matched audio campaign and summarize every sample."""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, math, statistics, tempfile


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_file(path, spec):
    assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], str(path)


def check_result(actual, expected, path=''):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        return max([check_result(actual[k], v, path+'/'+k) for k, v in expected.items()]+[0.])
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        return max([check_result(a, e, path+'/'+str(i)) for i, (a, e) in enumerate(zip(actual, expected))]+[0.])
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)) and math.isfinite(actual), path
        error = abs(actual-expected)
        if '/centroid/' in path:
            error /= max(1, abs(expected))
            assert error <= 1e-4, path
            return error
        assert error <= (1e-12 if path.startswith(('/intervals/', '/exclusive_intervals/')) else 0), path
    else:
        assert type(actual) == type(expected) and actual == expected, path
    return 0.


def validate_worker(result, manifest, mode):
    assert result['schema'] == 1 and result['family'] == manifest['family']
    assert result['conformance'] == (mode == 'conformance')
    assert result['affinity'] == 4 and result['held_outputs_unchanged'] is True
    assert math.isfinite(result['setup_seconds']) and result['setup_seconds'] > 0
    assert not any(k.lower().startswith(('lokad_', 'dotnet_', 'complus_')) for k in result['flags'])
    passes = 1 if mode == 'conformance' else 4
    expected = [(p, c) for p in range(passes) for c in manifest['cases']]
    assert len(result['records']) == len(expected), 'Incomplete sample coverage'
    first = {}
    previous_end = 0
    for row, (iteration, case) in zip(result['records'], expected):
        assert row['name'] == case['name'] and row['pass'] == iteration
        assert row['phase'] == ('warmup' if iteration == 0 else 'measured')
        assert row['ownership'] is True and row['input_sha256'] == case['raw_sha256']
        assert row['frequency'] > 0 and row['end_ticks'] > row['start_ticks']
        assert row['start_ticks'] >= previous_end
        previous_end = row['end_ticks']
        assert math.isfinite(row['seconds']) and row['seconds'] > 0
        assert math.isclose(row['seconds'], (row['end_ticks']-row['start_ticks'])/row['frequency'], rel_tol=1e-14)
        error = check_result(row['result'], case['expected'])
        assert math.isclose(row['maximum_centroid_error'], error, rel_tol=1e-12, abs_tol=1e-15)
        if case['name'] in first:
            assert row['result'] == first[case['name']], 'Repeat output changed'
        first[case['name']] = row['result']


def audit(root, base):
    import numpy as np
    spec = importlib.util.spec_from_file_location('process_accounting', root/'eng/campaign_processes.py')
    accounting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accounting)
    manifests = {}
    for family in ('parakeet', 'pyannote'):
        manifest = read(base/'inputs'/(family+'.json'))
        assert manifest['warmup_passes'] == 1 and manifest['measured_passes'] == 3
        upstream = [manifest['upstream']] if family == 'parakeet' else list(manifest['upstream'].values())
        specs = list(manifest['models'].values()) + [manifest['reference']] + upstream + list(manifest.get('native_assets', {}).values())
        specs += [c['pcm'] for c in manifest['cases']]
        for spec in specs:
            path = root/spec['path']
            verify_file(path, spec)
        for case in manifest['cases']:
            pcm = np.load(root/case['pcm']['path'], allow_pickle=False)
            assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
            case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
        manifests[family] = manifest
    frozen = read(base/'conformance/frozen.json')
    timing_frozen = read(base/'timing/frozen.json')
    assert sha(base/'conformance/prospective-plan.md') == frozen['plan_sha256']
    assert sha(base/'timing/prospective-plan.md') == timing_frozen['plan_sha256']
    assert frozen['files'] == timing_frozen['files']
    assert timing_frozen['conformance_identity_sha256'] == sha(base/'conformance/identity.json')
    for rel, spec in frozen['files'].items():
        verify_file(root/rel, spec)
    workers = []
    native_identity = None
    conformance_example = None
    for mode in ('conformance', 'timing'):
        directory = base/mode
        identity = read(directory/'identity.json')
        assert identity['complete'] is True and 'error' not in identity
        assert identity['supervisor_affinity'] == [0]
        sequence = [(family, engine) for family in ('parakeet', 'pyannote') for engine in (('ort', 'managed') if mode == 'conformance' else ('managed', 'ort', 'ort', 'managed'))]
        assert len(identity['runs']) == len(sequence)
        previous_end = identity['started']
        for index, (job, (family, engine)) in enumerate(zip(identity['runs'], sequence)):
            assert (job['index'], job['family'], job['engine']) == (index, family, engine)
            assert job['code'] == 0 and job['started'] >= previous_end and job['ended'] > job['started']
            assert job['samples'] and job['members'] and job['accounting']['valid']
            assert job['peak_rss'] == max(s['rss'] for s in job['samples']) < 20*1024**3
            assert job['seconds'] < 3600
            before = read(directory/(job['name']+'-pre.json'))
            after = read(directory/(job['name']+'-post.json'))
            assert accounting.foreign_fraction(before, after, identity['supervisor']) == job['accounting']
            assert not any(p['id'] == job['pid'] for p in after['processes']), 'Worker still present'
            previous_end = job['ended']
            for sample in job['samples']:
                assert sample['rss'] == sum(m['rss'] for m in sample['members'])
                for member in sample['members']:
                    assert member['affinity'] == [2]
                    assert job['members'][str(member['pid'])]['create_time'] == member['create_time']
            result = read(directory/job['name']/'result.json')
            assert result['engine'] == engine
            assert result['manifest_sha256'] == sha(base/'inputs'/(family+'.json'))
            validate_worker(result, manifests[family], mode)
            assert sum(r['seconds'] for r in result['records']) + result['setup_seconds'] <= job['seconds']
            for i, row in enumerate(result['records']):
                assert read(directory/job['name']/f'{i:03d}.json') == row
            if engine == 'managed':
                assert result['flags'] == {} and result['processor_count'] == 1
                for name, key in [('Lokad.Onnx.dll', 'core_sha256'), ('Lokad.Onnx.Data.dll', 'data_sha256'), ('AudioBenchmark.dll', 'runner_sha256')]:
                    assert result[key] == sha(base/'bin'/name)
                assert result['runtime'] == '.NET 10.0.12'
            else:
                assert result['onnxruntime'] == '1.29.0' and result['numpy'] == '2.2.4'
                assert result['native_settings'] == dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1, execution='sequential', optimization='all', spinning=False)
                assert all(result['flags'][key] == '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
                assert result['runner_sha256'] == sha(root/'tests/audio/comparison/native.py')
                assert result['adapter_sha256'] == sha(root/'tests/audio/comparison/native_adapters.py')
                assert result['python_binary_sha256'] == sha(Path(job['command'][0]))
                for path, digest in result['native_binaries'].items():
                    assert sha(Path(path)) == digest
                if native_identity is not None:
                    assert result['native_binaries'] == native_identity
                native_identity = result['native_binaries']
            if conformance_example is None:
                conformance_example = (result, manifests[family], mode)
            workers.append(dict(mode=mode, family=family, engine=engine, name=job['name'], setup_seconds=result['setup_seconds'], peak_rss=job['peak_rss'], foreign_cpu_fraction=job['accounting']['foreign_cpu_fraction'], maximum_centroid_error=max(r['maximum_centroid_error'] for r in result['records']), records=result['records']))
    timing = [w for w in workers if w['mode'] == 'timing']
    summaries = []
    for family, manifest in manifests.items():
        scopes = [('all_clips', manifest['cases'])] if family == 'parakeet' else [(c['name'], [c]) for c in manifest['cases']]
        for name, cases in scopes:
            names = {c['name'] for c in cases}
            seconds = sum(c['samples'] for c in cases)/16000
            row = dict(family=family, workload=name, audio_seconds=seconds, clips=len(cases), engines={})
            for engine in ('managed', 'ort'):
                visits = [w for w in timing if w['family'] == family and w['engine'] == engine]
                times = [[r['seconds'] for r in w['records'] if r['phase'] == 'measured' and r['name'] in names] for w in visits]
                flat = sum(times, [])
                visit_means = [sum(t)/3 for t in times]
                average = sum(flat)/6
                row['engines'][engine] = dict(measured_requests=len(flat), mean_corpus_seconds=average, mean_request_seconds=statistics.mean(flat), median_request_seconds=statistics.median(flat), minimum_request_seconds=min(flat), maximum_request_seconds=max(flat), rtf=average/seconds, visit_corpus_seconds=visit_means, visit_ratio=max(visit_means)/min(visit_means), setup_seconds=[w['setup_seconds'] for w in visits], peak_rss=[w['peak_rss'] for w in visits])
            row['managed_over_ort'] = row['engines']['managed']['mean_corpus_seconds']/row['engines']['ort']['mean_corpus_seconds']
            summaries.append(row)
    refusal_checks(*conformance_example)
    request_counts = {mode:sum(len(w['records']) for w in workers if w['mode'] == mode) for mode in ('conformance', 'timing')}
    assert request_counts == {'conformance':48, 'timing':384}
    assert sum(r['phase']=='measured' for w in timing for r in w['records']) == 288
    files = {p.relative_to(base).as_posix():dict(sha256=sha(p), bytes=p.stat().st_size) for folder in ('conformance', 'timing', 'inputs') for p in sorted((base/folder).rglob('*')) if p.is_file()}
    compact = []
    for worker in workers:
        w = dict(worker)
        w['records'] = [{k:r[k] for k in ('name', 'pass', 'phase', 'seconds', 'input_sha256', 'maximum_centroid_error')} for r in worker['records']]
        compact.append(w)
    return dict(schema=1, product_source=frozen['product_source'], frozen_files=frozen['files'], native_binaries=native_identity, request_counts=request_counts, measured_requests=288, refusal_checks=9, summaries=summaries, workers=compact, evidence_files=files)


def refusal_checks(result, manifest, mode):
    for change in (
        lambda r:r['records'].pop(),
        lambda r:r.update(affinity=8),
        lambda r:r['records'][0].update(input_sha256='0'*64),
        lambda r:r['records'][0].update(seconds=-1),
        lambda r:r['records'][0]['result']['token_ids'].append(999),
        lambda r:r['records'][0].update(ownership=False),
        lambda r:r['records'][0].update(phase='measured'),
    ):
        damaged = copy.deepcopy(result)
        change(damaged)
        try:
            validate_worker(damaged, manifest, mode)
        except AssertionError:
            continue
        raise AssertionError('Damaged evidence accepted')
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'binary.dll'
        path.write_bytes(b'original')
        pin = dict(bytes=8, sha256=sha(path))
        verify_file(path, pin)
        for damaged in (b'modified', b'truncated'):
            path.write_bytes(damaged)
            try:
                verify_file(path, pin)
            except AssertionError:
                continue
            raise AssertionError('Altered binary accepted')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    result = audit(a.root.resolve(), a.artifact.resolve())
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['summaries'], indent=2))

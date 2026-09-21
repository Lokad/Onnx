"""Retain the original pyannote graph/public checks on the corrected Core."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))
from common import ROOT, BASE as MODEL, pin, read, save, verify, terminal, psutil, monitor, rel
import numpy as np

BASE = ROOT / 'artifacts/parakeet-reduction-pyannote-20260921'
OLD = ROOT / 'artifacts/pyannote-lstm-output-lanes-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
CORE = 'f2c292cb6856e7ec80769e983df1f01512e98ad8d4d6d6ec3d424c6028e8f791'
PROFILE = '3f40a0c90ca2eaa0d19b96a8ca41bb97c4da5c11484a7549ef15346dfb493b7b'
SUITES = ROOT / 'artifacts/parakeet-reduction-dispatch-suites-20260921'


def prepare():
    proof = read(MODEL / 'closed.json')
    assert proof['evidence_passed'] and proof['native_numeric_passed']
    verify(proof['files'])
    for identity in proof['terminal_identities']:
        terminal(identity)
    suites = read(SUITES / 'closed.json')
    assert suites['passed']
    verify(suites['files'])
    for identity in [suites['supervisor'], *suites['worker_identities']]:
        terminal(identity)
    assert pin(OLD / 'qualification-closed.json')['sha256'] == 'cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    prior = read(OLD / 'qualification-closed.json')
    assert prior['passed']
    historical = {Path(k).as_posix(): v for k, v in prior['files'].items()}
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    runtime = BASE / 'runtime'
    runtime.mkdir()
    files = {}
    def bind(p, previous=False):
        value = pin(p)
        if previous:
            assert value == historical[rel(p)], str(p)
        files[rel(p)] = value
    for p in (OLD / 'runtimes/baseline').iterdir():
        if p.is_file():
            bind(p, True)
            shutil.copy2(p, runtime / p.name)
    assert pin(runtime / 'Profile.dll')['sha256'] == PROFILE
    assert pin(runtime / 'Lokad.Onnx.Data.dll') == pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')
    shutil.copy2(MODEL / 'runtime/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')
    assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE
    for p in runtime.iterdir():
        if p.is_file(): bind(p)
    bind(INPUT, True)
    spec = read(INPUT)
    for item in [*spec['models'].values(), spec['reference'], *[c['pcm'] for c in spec['cases']]]:
        p = ROOT / item['path']
        assert pin(p) == {k: item[k] for k in ('bytes', 'sha256')}
        bind(p, True)
    reference = ROOT / spec['reference']['path']
    native = read(reference)
    for case in native['cases']:
        if case['name'] not in {c['name'] for c in spec['cases'] if c['samples'] == 160000}: continue
        for key in ('scores', 'encoded'):
            name = case['windows'][0][key]
            p = reference.parent / name
            assert pin(p) == {k: native['files'][name][k] for k in ('bytes', 'sha256')}
            bind(p, True)
    baseline = ROOT / 'artifacts/pyannote-performance-profile-20260921/output/result.json'
    bind(baseline, True)
    for p in (MODEL / 'closed.json', SUITES / 'closed.json', OLD / 'qualification-closed.json', Path(__file__),
              ROOT / 'tests/parakeet/reduction-dispatch/common.py', ROOT / 'tests/parakeet/packing-budgets/common.py'):
        bind(p)
    save(BASE / 'prepared.json', dict(passed=True, files=files, core=CORE, profile=PROFILE))
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), files=len(files))))


def run():
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    assert not (BASE / 'processes.json').exists()
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    monitor.BASE = BASE
    try:
        monitor.worker(state, BASE / 'processes.json', 'pyannote',
            ['dotnet', BASE / 'runtime/Profile.dll', ROOT, INPUT, BASE / 'output', CORE],
            ROOT, [0], 10, 8, 900, False, BASE / 'output')
        verify(spec['files'])
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


def error(left, right):
    assert left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64)) / np.maximum(1., np.abs(right.astype(np.float64)))
    maximum = float(delta.max(initial=0))
    assert not (delta > 1e-4).any()
    return maximum


def audit():
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    state = read(BASE / 'processes.json')
    assert state['complete'] and state['code'] == 0 and len(state['runs']) == 1
    row = state['runs'][0]
    identities = [state['supervisor']] + [dict(pid=int(pid), birth=birth) for pid, birth in row['members'].items()]
    for identity in identities: terminal(identity)
    assert row['complete'] and row['code'] == 0 and row['preflight']['available'] >= 10 * 1024**3
    samples = [json.loads(s) for s in (BASE / 'logs/pyannote.samples.jsonl').read_text().splitlines()]
    assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
    assert all(s['seconds'] < 900 and s['rss'] < 8 * 1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
               and s['output_bytes'] <= 1024**3 and len(s['members']) <= 1 and all(p['affinity'] == [2] for p in s['members']) for s in samples)
    manifest = read(INPUT)
    reference = ROOT / manifest['reference']['path']
    native = read(reference)
    native_cases = {c['name']: c for c in native['cases']}
    result = read(BASE / 'output/result.json')
    assert result['passed'] and result['inputs_and_held_outputs_unchanged'] and result['runtime'] == '10.0.12'
    assert result['core_sha256'] == CORE and result['data_sha256'] == pin(MODEL / 'runtime/Lokad.Onnx.Data.dll')['sha256']
    assert result['manifest_sha256'] == pin(INPUT)['sha256']
    crops = [c['name'] for c in manifest['cases'] if c['samples'] == 160000]
    assert [(r['name'], r['model'], r['pass']) for r in result['rows']] == [(c, m, p) for c in crops for m in ('segmentation', 'embedding') for p in range(3)]
    baseline = read(ROOT / 'artifacts/pyannote-performance-profile-20260921/output/result.json')
    arrays, files, groups = [], set(), {}
    for actual, old in zip(result['rows'], baseline['rows'], strict=True):
        assert (actual['name'], actual['model'], actual['pass']) == (old['name'], old['model'], old['pass'])
        for kind in ('input', 'output'):
            info = actual[kind]
            p = BASE / 'output' / info['file']
            values = np.fromfile(p, dtype='<f4').reshape(info['shape'])
            assert pin(p)['sha256'] == info['sha256'] and values.size == info['values'] and np.isfinite(values).all()
            files.add(info['file'])
            if kind == 'input': assert info['sha256'] == old[kind]['sha256']
        name = native_cases[actual['name']]['windows'][0]['scores' if actual['model'] == 'segmentation' else 'encoded']
        wanted = np.load(reference.parent / name, allow_pickle=False)
        maximum = error(values, wanted)
        arrays.append(dict(name=actual['name'], model=actual['model'], pass_index=actual['pass'], values=values.size,
                           maximum=maximum, changed_bits=actual['output']['sha256'] != old['output']['sha256']))
        groups.setdefault((actual['name'], actual['model']), set()).add(actual['output']['sha256'])
    assert len(files) == 24 and {p.name for p in (BASE / 'output').iterdir()} == files | {'result.json'}
    assert all(len(v) == 1 for v in groups.values())
    assert [(a['name'], a['pass'], a['phase']) for a in result['applications']] == [(c['name'], p, 'warmup' if p == 0 else 'measured') for p in range(4) for c in manifest['cases']]
    maximum, first = 0., {}
    for item in result['applications']:
        a = item['result']
        e = next(c['expected'] for c in manifest['cases'] if c['name'] == item['name'])
        assert a['Status'] == 0 and e['status'] == 'Completed' and a['Windows'] == e['windows'] and a['AudioDuration'] == e['audio_seconds']
        for left, right in (('Intervals', 'intervals'), ('ExclusiveIntervals', 'exclusive_intervals')):
            for x, y in zip(a[left], e[right], strict=True):
                assert x['Speaker'] == y[2] and abs(x['Start'] - y[0]) <= 1e-12 and abs(x['End'] - y[1]) <= 1e-12
        local = 0.
        for x, y in zip(a['Speakers'], e['speakers'], strict=True):
            assert x['Speaker'] == y['speaker'] and x['HasEmbedding'] == y['has_embedding']
            assert len(x['Centroid']) == len(y['centroid']) == 256
            local = max(local, error(np.array(x['Centroid']), np.array(y['centroid'])))
        assert local == item['maximum_centroid_error']
        maximum = max(maximum, local)
        if item['name'] in first: assert a == first[item['name']]
        else: first[item['name']] = a
    analysis = dict(passed=True, arrays=arrays, values=sum(r['values'] for r in arrays), public_requests=len(result['applications']),
                    maximum_centroid_error=maximum, resources=dict(samples=len(samples), peak_rss=row['peak_rss']),
                    terminal_identities=identities, scope='Pyannote regression and complete native checks; no timing comparison')
    assert not (BASE / 'analysis.json').exists() and not (BASE / 'closed.json').exists()
    save(BASE / 'analysis.json', analysis)
    files = dict(spec['files'])
    files.update({rel(p): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE / 'closed.json', dict(passed=True, files=files, terminal_identities=identities))
    print(json.dumps(dict(passed=True, arrays=len(arrays), changed_arrays=sum(r['changed_bits'] for r in arrays),
                         values=analysis['values'], public_requests=analysis['public_requests'], maximum_centroid_error=maximum,
                         resources=analysis['resources'], closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run', 'audit')
    globals()[sys.argv[1]]()

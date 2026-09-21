"""Original full shared-model consumer for the corrected tensor dispatch."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))
from common import ROOT, BASE as MODEL, pin, read, save, verify, terminal, psutil, monitor, rel
import numpy as np

BASE = ROOT / 'artifacts/parakeet-reduction-shared-v2-20260921'
OLD = ROOT / 'artifacts/e5-profiler-shared-v2-20260921'
E5 = ROOT / 'artifacts/e5-randomized-processes-20260921/payload/inputs'
REFERENCE = ROOT / 'artifacts/shared-regression-20260918/reference'
CASES = ['e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok']
CORE = 'f2c292cb6856e7ec80769e983df1f01512e98ad8d4d6d6ec3d424c6028e8f791'
REPLAY = 'a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'


def prepare():
    proof = read(MODEL / 'closed.json')
    assert proof['evidence_passed'] and proof['native_numeric_passed']
    verify(proof['files'])
    for identity in proof['terminal_identities']:
        terminal(identity)
    assert pin(OLD / 'closed.json')['sha256'] == '77f2ab0b4761b09be7225b69667e071a05826fb0c053a67b1e6d2a93a99ea533'
    prior = read(OLD / 'closed.json')
    assert prior['qualified']
    for identity in prior['identities']:
        terminal(identity)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    runtime = BASE / 'runtime'
    runtime.mkdir()
    files = {}
    def bind(p, historical=False):
        value = pin(p)
        if historical:
            assert value == prior['files'][rel(p)], str(p)
        files[rel(p)] = value
    for p in (OLD / 'runtimes/baseline').iterdir():
        if p.is_file():
            bind(p, True)
            shutil.copy2(p, runtime / p.name)
    assert pin(runtime / 'Replay.dll')['sha256'] == REPLAY
    # The original shared-model consumer depends only on Core; preserve its runtime.
    assert not (runtime / 'Lokad.Onnx.Data.dll').exists()
    shutil.copy2(MODEL / 'runtime/Lokad.Onnx.dll', runtime / 'Lokad.Onnx.dll')
    assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE
    for p in runtime.iterdir():
        if p.is_file():
            bind(p)
    bind(REFERENCE / 'manifest.json', True)
    for model in read(REFERENCE / 'manifest.json')['models']:
        for asset in model['assets']:
            p = ROOT / asset['file']
            assert pin(p) == {k: asset[k] for k in ('bytes', 'sha256')}
            bind(p, True)
        for scenario in model['scenarios']:
            for step in scenario['steps']:
                for record in step['inputs'] + step['outputs']:
                    p = REFERENCE / record['file']
                    assert pin(p)['sha256'] == record['sha256']
                    bind(p, True)
    for name in CASES:
        p = E5 / (name + '.json')
        bind(p, True)
        fixture = read(p)
        reference = E5 / fixture['reference_file']
        assert pin(reference)['sha256'] == fixture['reference_sha256']
        assert pin(ROOT / 'models/multilingual-e5-small/model.onnx')['sha256'] == fixture['model_sha256']
        bind(reference, True)
    bind(ROOT / 'models/multilingual-e5-small/model.onnx', True)
    for mode in ('shared', 'e5'):
        bind(OLD / 'outputs' / (mode + '-0-baseline') / 'result.json', True)
    for p in (MODEL / 'closed.json', OLD / 'closed.json', Path(__file__),
              ROOT / 'artifacts/parakeet-reduction-shared-20260921/failure-closed.json',
              ROOT / 'tests/parakeet/reduction-shared/qualify.py',
              ROOT / 'tests/parakeet/reduction-dispatch/common.py',
              ROOT / 'tests/parakeet/packing-budgets/common.py'):
        bind(p)
    save(BASE / 'prepared.json', dict(passed=True, core=CORE, replay=REPLAY, files=files,
         jobs=['shared', 'e5'], limits=dict(preflight_gib=10, rss_gib=8, seconds=900)))
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), files=len(files))))


def run():
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    assert not (BASE / 'processes.json').exists()
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    monitor.BASE = BASE
    old_env = monitor.clean_env
    monitor.clean_env = lambda: dict(old_env(), LOKAD_ONNX_FINGERPRINT_STRINGS='0')
    try:
        for mode in spec['jobs']:
            reference = REFERENCE if mode == 'shared' else E5
            folder = BASE / 'outputs' / mode
            monitor.worker(state, BASE / 'processes.json', mode,
                ['dotnet', BASE / 'runtime/Replay.dll', mode, ROOT, reference, folder, CORE],
                ROOT, [0], 10, 8, 900, False, BASE / 'outputs')
            result = read(folder / 'result.json')
            assert result['passed'] and result['core_sha256'] == CORE
            print(mode, 'complete arrays', len(result['rows']), flush=True)
        verify(spec['files'])
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        monitor.clean_env = old_env
        state['complete'] = True
        save(BASE / 'processes.json', state)


def expected_rows(mode):
    if mode == 'e5':
        return [(case, policy + '-' + context, step, 'last_hidden_state', read(E5 / (case + '.json'))['reference_file'])
                for case in CASES for policy in ('default', 'memory') for context in ('facade', 'context') for step in range(3)]
    return [(model['key'], scenario['name'], step, value['name'], value['file'])
            for model in read(REFERENCE / 'manifest.json')['models'] for scenario in model['scenarios']
            for step, item in enumerate(scenario['steps']) for value in item['outputs']]


def fnv(values):
    result = 1469598103934665603
    for value in values.view('<u4').reshape(-1):
        result = ((result ^ int(value)) * 1099511628211) & ((1 << 64) - 1)
    return result


def audit():
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    state = read(BASE / 'processes.json')
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == spec['jobs']
    identities = [state['supervisor']]
    rows, resources, hashes = [], [], []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['preflight']['available'] >= 10 * 1024**3
        identities.extend(dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items())
        samples = [json.loads(s) for s in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == run['samples'] > 0 and max(s['rss'] for s in samples) == run['peak_rss']
        assert all(s['seconds'] < 900 and s['rss'] < 8 * 1024**3 and s['available'] >= 1024**3 and s['disk'] >= 20 * 1024**3
                   and s['output_bytes'] <= 1024**3 and len(s['members']) <= 1
                   and all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in s['members']) for s in samples)
        resources.append(dict(name=run['name'], samples=len(samples), peak_rss=run['peak_rss']))
        mode = run['name']
        folder = BASE / 'outputs' / mode
        result = read(folder / 'result.json')
        assert result['passed'] and result['mode'] == mode and result['enabled'] is False
        assert result['core_sha256'] == CORE and result['probe_sha256'] == REPLAY
        assert result['runtime'] == '10.0.12' and result['flags'] == dict(LOKAD_ONNX_FINGERPRINT_STRINGS='0')
        assert result['inputs_unchanged'] and result['held_outputs_unchanged']
        assert len(result['graphs']) == (80 if mode == 'e5' else 11) and all(g['entries'] == 0 for g in result['graphs'])
        assert [(r['model'], r['scenario'], r['step'], r['name'], r['reference_file']) for r in result['rows']] == expected_rows(mode)
        prior = read(OLD / 'outputs' / (mode + '-0-baseline') / 'result.json')
        reference = E5 if mode == 'e5' else REFERENCE
        assert {p.name for p in folder.iterdir()} == {'result.json'} | {str(i) + '.f32' for i in range(len(result['rows']))}
        for index, (row, old) in enumerate(zip(result['rows'], prior['rows'], strict=True)):
            assert all(row[k] == old[k] for k in ('model', 'scenario', 'step', 'name', 'shape', 'values', 'reference_file', 'reference_sha256'))
            p = folder / row['file']
            assert row['file'] == str(index) + '.f32' and pin(p)['sha256'] == row['sha256']
            native = reference / row['reference_file']
            assert pin(native)['sha256'] == row['reference_sha256']
            actual = np.fromfile(p, dtype='<f4')
            wanted = np.fromfile(native, dtype='<f4') if mode == 'e5' else np.load(native, allow_pickle=False)
            assert actual.dtype == wanted.dtype == np.float32 and np.isfinite(actual).all() and np.isfinite(wanted).all()
            assert actual.size == wanted.size == row['values'] == int(np.prod(row['shape']))
            assert row['shape'] == (read(E5 / (row['model'] + '.json'))['shape'] if mode == 'e5' else list(wanted.shape))
            delta = np.abs(actual.astype(np.float64) - wanted.reshape(-1).astype(np.float64)) / np.maximum(1., np.abs(wanted.reshape(-1).astype(np.float64)))
            maximum, failed = float(delta.max(initial=0)), int(np.count_nonzero(delta > 1e-4))
            assert failed == row['failed_values'] == 0 and abs(maximum - row['max_scaled_error']) <= 1e-15
            rows.append(dict(mode=mode, model=row['model'], scenario=row['scenario'], step=row['step'], name=row['name'],
                             values=actual.size, maximum=maximum, changed_bits=row['sha256'] != old['sha256']))
            if row['model'] == 'dinov3':
                hashes.append(dict(scenario=row['scenario'], step=row['step'], name=row['name'], hash=fnv(actual),
                                   maximum=maximum, values=actual.size, sha256=row['sha256']))
    for identity in identities:
        terminal(identity)
    assert len(rows) == 166 and sum(r['values'] for r in rows) == 5000814
    analysis = dict(passed=True, arrays=len(rows), values=sum(r['values'] for r in rows), rows=rows, dino_hashes=hashes,
                    resources=resources, terminal_identities=identities, scope='Full native shared outputs, no timing or production promotion')
    assert not (BASE / 'analysis.json').exists() and not (BASE / 'closed.json').exists()
    save(BASE / 'analysis.json', analysis)
    files = dict(spec['files'])
    files.update({rel(p): pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE / 'closed.json', dict(passed=True, files=files, terminal_identities=identities, analysis=pin(BASE / 'analysis.json')))
    print(json.dumps(dict(passed=True, arrays=len(rows), values=analysis['values'], dino_hashes=hashes,
                         changed_arrays=sum(r['changed_bits'] for r in rows), resources=resources, closed=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run', 'audit')
    globals()[sys.argv[1]]()

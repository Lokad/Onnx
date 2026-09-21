"""Full affected Parakeet trajectory; preserve all original native limitations."""
import importlib.util
import shutil
import sys
import traceback
from common import *

BASE = ROOT / 'artifacts/pyannote-convolution-portable-parakeet-20260921'
PRIOR = ROOT / 'artifacts/pyannote-optimized-parakeet-20260921'
FROZEN = ROOT / 'artifacts/parakeet-transcription-20260919/frozen'
REFERENCE = FROZEN / 'reference/manifest.json'
MODELS = ROOT / 'models/parakeet-tdt-0.6b-v3'
AUDITOR = ROOT / 'tests/parakeet/transcribe/audit.py'
REPLAY = '335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
KNOWN = ['english-16k/step-26/outputs', 'english-frame-limit/step-26/outputs', 'english-repeat/step-26/outputs']


def prepare():
    candidate()
    assert pin(PRIOR / 'closed.json')['sha256'] == '57ded6ece06f486c31689ef14143237a0d188bcd76ced094a39932a45c406817'
    prior = read(PRIOR / 'closed.json')
    assert prior['regression_passed'] and not prior['candidate_native_numeric_passed']
    verify(prior['files'])
    for identity in prior['identities']:
        terminal(identity)
    assert pin(REFERENCE)['sha256'] == '3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    runtime = BASE / 'runtime'
    shutil.copytree(PRIOR / 'runtimes/candidate', runtime)
    for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']:
        shutil.copy2(MODEL / 'runtime' / name, runtime / name)
    assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == CORE
    assert pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'] == DATA
    assert pin(runtime / 'TranscribeReplay.dll')['sha256'] == REPLAY
    files = dict(prior['files'])
    for path in [MODEL / 'closed.json', PRIOR / 'closed.json', AUDITOR, MONITOR, Path(__file__), TOOLS / 'common.py']:
        files[rel(path)] = pin(path)
    for path in runtime.iterdir():
        if path.is_file():
            files[rel(path)] = pin(path)
    save(BASE / 'prepared.json', dict(passed=True, files=files, core=CORE, data=DATA, runner=REPLAY,
        known_native_failures=KNOWN, limits=dict(preflight_gib=10, rss_gib=8, seconds=1800),
        criterion='All 784 output arrays bit-identical to qualified predecessor; original native audit and three failures preserved.'))
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'))))


def run():
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    assert not (BASE / 'processes.json').exists()
    own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    monitor.BASE = BASE
    try:
        row = monitor.worker(state, BASE / 'processes.json', 'parakeet',
            ['dotnet', BASE / 'runtime/TranscribeReplay.dll', MODELS, REFERENCE, BASE / 'candidate.json'],
            ROOT, [1], 10, 8, 1800, False, BASE / 'candidate.json.tensors')
        result = read(BASE / 'candidate.json')
        assert result['application_passed'] and not result['errors'] and not result['passed']
        assert result['comparisons'] == 784 and result['values_compared'] == 3090494
        verify(spec['files'])
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(BASE / 'processes.json', state)


def audit():
    spec = read(BASE / 'prepared.json')
    verify(spec['files'])
    resource = resources(BASE, 'processes.json', {'parakeet': (10, 1800, True)})
    module_spec = importlib.util.spec_from_file_location('original_transcription_auditor', AUDITOR)
    auditor = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(auditor)
    report = auditor.audit(REFERENCE, BASE / 'candidate.json')
    previous_report = auditor.audit(REFERENCE, PRIOR / 'candidate.json')
    assert report['audit_consistent'] and report['application_passed'] and not report['numeric_gate_passed']
    assert report['failures'] == previous_report['failures']
    assert ['/'.join(r[k] for k in ('case', 'label', 'output')) for r in report['failures']] == KNOWN
    result, old = read(BASE / 'candidate.json'), read(PRIOR / 'candidate.json')
    assert result['core_sha256'] == CORE and result['data_sha256'] == DATA and result['runner_sha256'] == REPLAY
    assert result['runtime'] == '.NET 10.0.12' and not result['settings']
    count = values = 0
    for row, prior in zip(result['rows'], old['rows'], strict=True):
        assert row['name'] == prior['name'] and row['actual'] == prior['actual']
        for a, b in zip(row['comparisons'], prior['comparisons'], strict=True):
            assert a == b, (row['name'], a['label'], a['output'])
            left = BASE / 'candidate.json.tensors' / a['file']
            right = PRIOR / 'candidate.json.tensors' / b['file']
            assert pin(left) == pin(right)
            count += 1
            values += pin(left)['bytes'] // (8 if a['dtype'] == 'Int64' else 4)
    assert count == 784 and values == 3090494
    analysis = dict(passed=True, arrays=count, values=values, bit_identical_arrays=count, native=report,
        **resource, scope='Affected Parakeet application regression; same three native numerical failures; no speed claim.')
    close(BASE, analysis, spec['files'], resource['identities'])
    print(json.dumps(dict(passed=True, arrays=count, values=values, samples=resource['samples'], native_failures=report['failures'])))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run', 'audit')
    globals()[sys.argv[1]]()

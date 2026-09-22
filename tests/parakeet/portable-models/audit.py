"""Recompute every native output using the original independent validators."""
import hashlib
from common import *
import numpy as np


def main():
    prepared = read(BASE / 'prepared.json'); assert prepared['passed']; verify(prepared['files'])
    built = resources(BASE, 'build-state.json', {n: (8, 8, 900, False) for n in ['consumer-restore', 'consumer-build', 'consumer-instructions']})
    executed = resources(BASE, 'processes.json', {'native': (10, 8, 1200, True), 'public': (14, 12, 1200, True)})
    native_auditor = module('original_native_auditor', ROOT / 'tests/parakeet/transcribe/audit.py')
    native = native_auditor.audit(REFERENCE, BASE / 'native/result.json')
    assert native['numeric_gate_passed'] and native['application_passed'] and native['audit_consistent']
    assert native['arrays'] == 784 and native['values'] == 3090494 and not native['failures']
    result = read(BASE / 'native/result.json')
    assert result['core_sha256'] == CORE and result['data_sha256'] == DATA
    assert result['runner_sha256'] == pin(BASE / 'runtime/TranscribeReplay.dll')['sha256']
    assert result['runtime'] == '.NET 10.0.12' and not result['settings']
    public_auditor = module('original_public_auditor', ROOT / 'tests/audio/comparison/audit.py')
    manifest = read(CORPUS)
    for case in manifest['cases']:
        pcm = np.load(ROOT / case['pcm']['path'], allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    result = read(BASE / 'public/output/result.json')
    public_auditor.validate_worker(result, manifest, 'conformance')
    assert result['engine'] == 'managed' and result['runtime'] == '.NET 10.0.12' and result['flags'] == {} and result['processor_count'] == 1
    assert result['core_sha256'] == CORE and result['data_sha256'] == DATA
    assert result['runner_sha256'] == pin(BASE / 'runtime/AudioBenchmark.dll')['sha256'] and result['manifest_sha256'] == pin(CORPUS)['sha256']
    assert [p.name for p in sorted((BASE / 'public/output').glob('[0-9][0-9][0-9].json'))] == [f'{i:03}.json' for i in range(20)]
    for i, row in enumerate(result['records']):
        assert read(BASE / 'public/output' / f'{i:03}.json') == row
    # Reuse every numerical, input, repeat, public, and ownership assertion in
    # the previously qualified Pyannote auditor. Only runtime/artifact globals change.
    original = ROOT / 'tests/parakeet/reduction-shared/qualify_pyannote.py'
    source = original.read_text(encoding='utf8')
    old = "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))\nfrom common import ROOT, BASE as MODEL, pin, read, save, verify, terminal, psutil, monitor, rel"
    assert source.count(old) == 1
    source = source.replace(old, 'from common import ROOT, MODEL, pin, read, save, verify, terminal, psutil, monitor, rel')
    namespace = dict(__name__='original_pyannote_auditor', __file__=str(original))
    exec(compile(source, str(original), 'exec'), namespace)
    namespace.update(BASE=BASE / 'pyannote', MODEL=MODEL, CORE=CORE)
    namespace['audit']()
    pyannote = read(BASE / 'pyannote/analysis.json')
    assert len(pyannote['arrays']) == 18 and pyannote['public_requests'] == 16
    identities = built['identities'] + executed['identities'] + pyannote['terminal_identities']
    analysis = dict(passed=True, native=native, public_requests=20, pyannote=pyannote,
        core=CORE, data=DATA, consumer_methods=96, unchanged_consumer_methods=95,
        resources=built['resources'] + executed['resources'], identities=identities,
        scope='Complete composed audio correctness; no speed or production promotion')
    close(BASE, analysis, prepared['files'], identities)
    print(json.dumps(dict(passed=True, parakeet_arrays=784, parakeet_values=3090494, maximum=native['maximum'],
        public_requests=20, pyannote_arrays=18, pyannote_public_requests=16)), flush=True)


if __name__ == '__main__': main()

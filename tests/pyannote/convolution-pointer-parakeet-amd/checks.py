"""Require all AMD native comparisons and complete exact same-platform results."""
import copy
import hashlib
import numpy as np
from protocol import pin, read
from native_audit import audit as native_audit
from public_audit import validate_worker


def identity(result, role, spec, consumer):
    assert result['core_sha256'] == spec['identities'][role]['Lokad.Onnx.dll']['sha256']
    assert result['data_sha256'] == spec['identities'][role]['Lokad.Onnx.Data.dll']['sha256']
    assert result['runner_sha256'] == spec['consumers'][consumer]['sha256']
    assert result['runtime'] == '.NET 10.0.8'


def manifest_with_raw_hashes(base, role):
    manifest = copy.deepcopy(read(base/'manifests'/(role+'-parakeet.json')))
    assert len(manifest['cases']) == 20
    for case in manifest['cases']:
        path = base/'assets'/case['pcm']['path']
        assert pin(path) == {k: case['pcm'][k] for k in ['bytes', 'sha256']}
        pcm = np.load(path, allow_pickle=False)
        assert pcm.dtype == np.float32 and pcm.shape == (case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256'] = hashlib.sha256(pcm.tobytes()).hexdigest()
    return manifest


def exact_native(base):
    before = read(base/'selected-native/result.json'); after = read(base/'candidate-native/result.json')
    comparisons = []
    for a, b in zip(before['rows'], after['rows'], strict=True):
        assert a['name'] == b['name'] and a['actual'] == b['actual']
        for x, y in zip(a['comparisons'], b['comparisons'], strict=True):
            assert all(x[k] == y[k] for k in ['label', 'output', 'shape', 'dtype', 'file'])
            left = (base/'selected-native/result.json.tensors'/x['file']).read_bytes()
            right = (base/'candidate-native/result.json.tensors'/y['file']).read_bytes()
            assert left == right, (a['name'], x['label'], x['output'])
            comparisons.append(dict(case=a['name'], label=x['label'], output=x['output'],
                values=len(left)//(8 if x['dtype'] == 'Int64' else 4), sha256=hashlib.sha256(left).hexdigest(), bit_identical=True))
    assert len(comparisons) == 784 and sum(r['values'] for r in comparisons) == 3090494
    return comparisons


def qualify(base, name, spec):
    role, mode = name.split('-'); path = base/name/('result.json' if mode == 'native' else 'output/result.json')
    result = read(path)
    identity(result, role, spec, 'TranscribeReplay' if mode == 'native' else 'AudioBenchmark')
    if mode == 'native':
        assert not result['settings']
        report = native_audit(base/'parakeet-reference/manifest.json', path)
        assert report['audit_consistent'] and report['application_passed'] and report['numeric_gate_passed']
        assert report['arrays'] == 784 and report['values'] == 3090494 and not report['failures']
        if role == 'candidate': report['exact_selected_comparisons'] = exact_native(base)
        return dict(passed=True, native=report, no_performance_measurement=True)
    assert mode == 'public'
    manifest = manifest_with_raw_hashes(base, role)
    assert result['engine'] == 'managed' and not result['flags'] and result['processor_count'] == 1
    assert result['manifest_sha256'] == pin(base/'manifests'/(role+'-parakeet.json'))['sha256']
    validate_worker(result, manifest, 'conformance')
    assert {p.name for p in path.parent.iterdir()} == {'result.json'} | {f'{i:03}.json' for i in range(20)}
    for i, row in enumerate(result['records']): assert read(path.parent/f'{i:03}.json') == row
    if role == 'candidate':
        previous = read(base/'selected-public/output/result.json')
        assert [r['result'] for r in result['records']] == [r['result'] for r in previous['records']]
    return dict(passed=True, public_requests=20, complete_selected_results_exact=True if role == 'candidate' else None,
                result=pin(path), no_performance_measurement=True)

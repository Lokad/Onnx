"""Require all AMD native comparisons and complete same-platform results with bounded float differences."""
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


def compare_native(base, isa):
    from cross_numeric import compare_native as compare
    return compare(base, isa)


def qualify(base, name, spec):
    role, mode, isa = name.split('-'); path = base/name/('result.json' if mode == 'native' else 'output/result.json')
    assert isa in ['512','256']
    expected_flags = {} if isa=='512' else {'DOTNET_EnableAVX512':'0'}
    result = read(path)
    identity(result, role, spec, 'TranscribeReplay' if mode == 'native' else 'AudioBenchmark')
    if mode == 'native':
        assert result['settings'] == expected_flags
        report = native_audit(base/'parakeet-reference/manifest.json', path)
        assert report['audit_consistent'] and report['application_passed'] and report['numeric_gate_passed']
        assert report['arrays'] == 784 and report['values'] == 3090494 and not report['failures']
        if role == 'candidate': report['selected_comparisons'] = compare_native(base, isa)
        return dict(passed=True, native=report, no_performance_measurement=True)
    assert mode == 'public'
    manifest = manifest_with_raw_hashes(base, role)
    assert result['engine'] == 'managed' and result['flags'] == expected_flags and result['processor_count'] == 1
    assert result['manifest_sha256'] == pin(base/'manifests'/(role+'-parakeet.json'))['sha256']
    # The raw flags were checked exactly above. Preserve every original public
    # assertion while neutralizing only the prospectively allowed ISA override.
    audited_result = copy.deepcopy(result); audited_result['flags'] = {}
    validate_worker(audited_result, manifest, 'conformance')
    assert {p.name for p in path.parent.iterdir()} == {'result.json'} | {f'{i:03}.json' for i in range(20)}
    for i, row in enumerate(result['records']): assert read(path.parent/f'{i:03}.json') == row
    if role == 'candidate':
        previous = read(base/f'selected-public-{isa}/output/result.json')
        assert [r['result'] for r in result['records']] == [r['result'] for r in previous['records']]
    return dict(passed=True, public_requests=20, complete_selected_results_exact=True if role == 'candidate' else None,
                result=pin(path), no_performance_measurement=True)

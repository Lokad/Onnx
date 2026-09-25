"""Bind a diagnostic to the qualified isolated candidate without promoting it."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'artifacts/parakeet-slice-dense-conversion-source-20260924'
BUILD = ROOT / 'artifacts/parakeet-slice-dense-conversion-recovery-amd-20260924'
TESTS = ROOT / 'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924'
RELEASE = ROOT / 'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924'
STAGES = {name: ROOT / f'artifacts/parakeet-slice-dense-conversion-{name}-amd-20260925'
          for name in ['models', 'app', 'shared', 'pyannote', 'graphs']}
TENSORS = ROOT / 'artifacts/parakeet-slice-dense-conversion-isa-amd-20260924'
TEST = 'tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'
OVERRIDE = 'Lokad.Onnx.TensorSlice`1[T]::ToDenseTensor::Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def source_path(name):
    return TESTS / 'bundle/source' / name if name == TEST else SOURCE / 'source' / name


def candidate_inventory(core, data):
    """Project the inspected candidate side; do not manufacture any method body."""
    assert core['assembly'] == 'Lokad.Onnx.dll' and core['methods'] == core['unchanged_methods'] == 3253
    assert not core['differences'] and not core['removed'] and core['added'] == [OVERRIDE]
    assert set(core['candidate_methods']) == {OVERRIDE}
    assert core['effective_conversion_signature_equal']
    assert all(core['method_flags_after'][k] == v for k, v in core['method_flags_before'].items())
    methods = dict(core['normalized_methods'], **core['candidate_methods'])
    assert len(methods) == len(core['method_flags_after']) == 3254
    assert set(methods) == set(core['method_flags_after'])
    assert set(core['public_surface_after']) - set(core['public_surface']) == {
        'MEMBER Lokad.Onnx.TensorSlice`1 Method Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()',
        'MEMBER Lokad.Onnx.TensorSlice`1 Method Lokad.Onnx.DenseTensor`1[T] ToDenseTensor() FLAGS Public, Virtual, HideBySig'}
    assert not set(core['public_surface']) - set(core['public_surface_after'])
    assert data['assembly'] == 'Lokad.Onnx.Data.dll' and data['methods'] == data['unchanged_methods'] == 697
    assert data['public_surface_equal'] and not data['differences'] and not data['removed'] and not data['added']
    result = [dict(assembly=core['assembly'], before_sha256=core['after_sha256'],
                   public_surface=core['public_surface_after'], normalized_methods=methods,
                   method_flags_before=core['method_flags_after']),
              dict(assembly=data['assembly'], before_sha256=data['before_sha256'],
                   public_surface=data['public_surface'], normalized_methods=data['normalized_methods'],
                   method_flags_before=data['method_flags_before'])]
    return dict(inventory_complete=True, observations=result)


def qualify():
    source = read(SOURCE / 'prepared.json')
    assert source['passed'] and pin(SOURCE / 'prepared.json')['sha256'] == 'a9811c6369122eb8b53ab1d88d8329928a999d172d89786a1b393c3451e91e1e'
    assert len(source['before']) == 427 and len(source['source']) == 428
    assert source['root_release'] == pin(RELEASE / 'closed.json')
    assert pin(RELEASE / 'closed.json')['sha256'] == 'dd612e81a85f74aebe4371ca93e6f17216c6779b927caf66bf400f0710af0f8e'
    for name, wanted in source['before'].items():
        assert pin(ROOT / name) == wanted, ('Release source changed', name)
    assert not (ROOT / TEST).exists(), 'This lane must not integrate the candidate'
    for name, wanted in source['source'].items():
        assert pin(SOURCE / 'source' / name) == wanted, name
    correction = read(TESTS / 'build-review.json')
    assert correction['passed'] and pin(TESTS / 'build-review.json')['sha256'] == '0eeef7982a09859c8f791bc361d900ab04941ada148e5600527978732b5351d4'
    files = dict(source['source'])
    files[TEST] = pin(source_path(TEST))
    assert files[TEST]['sha256'] == '29e0bd37dda89e0819807e3dd613785971e2977b6a1de5e19a7fd5ecbaff6c25'
    build = read(BUILD / 'build-review.json')
    assert build['passed'] and pin(BUILD / 'build-review.json')['sha256'] == '349ae125453183b0b04202dd6408c789760e1748ec863fc51c44d81208f17cbd'
    native_inventory = BUILD / 'build-collected/inventory/instructions.json'
    release_inventory = RELEASE / 'collected/inventory/instructions.json'
    assert build['inventory'] == pin(native_inventory)
    release = read(RELEASE / 'closed.json')
    assert release['passed'] and release['files']['collected/inventory/instructions.json'] == pin(release_inventory)
    core = read(native_inventory)['observations'][0]
    release_rows = read(release_inventory)['observations']
    selected_core = next(r for r in release_rows if r['assembly'] == 'Lokad.Onnx.dll')
    assert core['before_sha256'] == selected_core['before_sha256']
    assert core['normalized_methods'] == selected_core['normalized_methods']
    assert core['method_flags_before'] == selected_core['method_flags_before']
    data = next(r for r in release_rows if r['assembly'] == 'Lokad.Onnx.Data.dll')
    baseline = candidate_inventory(core, data)
    evidence = [SOURCE / 'prepared.json', BUILD / 'build-review.json', native_inventory,
                TESTS / 'build-review.json', RELEASE / 'closed.json', release_inventory,
                TENSORS / 'closed.json', TENSORS / 'analysis.json']
    tensor = read(TENSORS / 'closed.json')
    assert tensor['passed'] and tensor['analysis'] == pin(TENSORS / 'analysis.json')
    tensor_analysis = read(TENSORS / 'analysis.json')
    assert tensor_analysis['compiled_review'] == pin(BUILD / 'build-review.json')
    assert tensor_analysis['corrected_test_review'] == pin(TESTS / 'build-review.json')
    assert [(r['mode'], r['passed'], r['skipped']) for r in tensor_analysis['suites']] == [('512', 395, 0), ('256', 395, 0)]
    analyses, proofs = {}, {}
    for name, folder in STAGES.items():
        proof = read(folder / 'closed.json')
        assert proof['passed']
        for file, wanted in proof['files'].items():
            assert pin(folder / file) == wanted, file
        analyses[name], proofs[name] = read(folder / 'analysis.json'), proof
        assert analyses[name]['passed']
        evidence.extend([folder / 'closed.json', folder / 'analysis.json'])
    product = analyses['models']['identities']['candidate']
    assert product == analyses['app']['identities']['candidate'] == analyses['pyannote']['identities']['candidate']
    assert proofs['app']['admitted'] and tensor_analysis['core'] == product['Lokad.Onnx.dll']
    assert core['after_sha256'] == product['Lokad.Onnx.dll']['sha256'] == '49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert data['before_sha256'] == product['Lokad.Onnx.Data.dll']['sha256'] == 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    assert pin(STAGES['graphs'] / 'closed.json')['sha256'] == 'e04e3a6a7434afd4af6bda1901c934004fc263db82a52f453f4321ca3f7a9fbd'
    assert not proofs['graphs']['admitted'] and not proofs['graphs']['all_controls_passed']
    failures = [(r['key'], c['role']) for r in analyses['graphs']['performance'] for c in r['controls'] if not c['passed']]
    assert failures == [('e5-8tok', 'candidate'), ('e5-512tok', 'candidate')]
    assert all(r['regression_passed'] for r in analyses['graphs']['performance'])
    return dict(source_files=files, product=product, inventory=baseline,
        evidence={p.relative_to(ROOT).as_posix(): pin(p) for p in evidence},
        diagnostic_only=True, release_admitted=False, failed_release_controls=[list(r) for r in failures])


if __name__ == '__main__':
    value = qualify()
    print(json.dumps(dict(source_files=len(value['source_files']), methods=[len(r['normalized_methods']) for r in value['inventory']['observations']],
        release_admitted=value['release_admitted'], failed_release_controls=value['failed_release_controls'])))

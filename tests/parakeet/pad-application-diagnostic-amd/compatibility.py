"""Reconcile exact retained binaries without rebuilding or running inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / 'artifacts'
SELECTED = ARTIFACTS / 'parakeet-winograd-baseline-amd-20260923'
QUALIFIED = ARTIFACTS / 'pyannote-winograd-product-root-amd-20260923'
CANDIDATE = ARTIFACTS / 'parakeet-pad-dispatch-build-amd-20260923'
SCREEN = ARTIFACTS / 'parakeet-pad-dispatch-screen-amd-20260923'
OBSERVER = ARTIFACTS / 'parakeet-managed-phase-amd-20260924'
CONSUMER = ARTIFACTS / 'parakeet-slice-materialization-profile-amd-20260924'
DATA_SHA = 'a2a0b4901ba8e4270b57e2b176d7a656e739886275ec66d0ca036807eea16e3e'
RUNNER_SHA = '38ab5c7e65aa00ace50e8704348830286047e0c96ebc0a04b66c6e54d063899c'


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def reconcile(root, candidate, observer):
    """Fail on any unreviewed method, implementation flag or public binding."""
    assert root['inventory_complete'] and candidate['inventory_complete'] and observer['inventory_complete']
    root_core, root_data = root['observations']
    core, data = candidate['observations']
    observed_data, = [r for r in observer['observations'] if r['assembly'] == 'Lokad.Onnx.Data.dll']
    for row, count in [(root_core, 3179), (root_data, 697), (data, 697)]:
        assert row['methods'] == row['unchanged_methods'] == count
        assert not row['removed'] and not row['added'] and not row['differences'] and not row['candidate_methods']
        assert row['public_surface_equal']
    assert root_core['after_sha256'] == core['before_sha256']
    assert root_data['after_sha256'] == data['before_sha256']
    for first, second in [(root_core, core), (root_data, data), (data, observed_data)]:
        assert first['normalized_methods'] == second['normalized_methods']
        assert first['public_surface'] == second['public_surface']
    # The older root inspector did not record implementation flags. Do not
    # infer them from equal IL; the new exact-binary inspector must check them.
    assert data['method_flags_before'] == data['method_flags_after'] == observed_data['method_flags_before']
    assert core['public_surface_equal'] and not core['removed']
    assert core['methods'] == 3179 and core['unchanged_methods'] == 3178
    assert len(core['added']) == 1 and '::PadDispatch::' in core['added'][0]
    assert len(core['differences']) == 1 and '::Pad::' in core['differences'][0]
    assert all(core['method_flags_after'][k] == v for k, v in core['method_flags_before'].items())
    assert observed_data['methods'] == 697 and observed_data['unchanged_methods'] == 696
    assert not observed_data['removed'] and len(observed_data['added']) == 50
    assert len(observed_data['differences']) == 1 and '::Execute::' in observed_data['differences'][0]
    assert observed_data['after_sha256'] == DATA_SHA
    assert all(observed_data['method_flags_after'][k] == v for k, v in observed_data['method_flags_before'].items())
    # Resolve each observer helper's public Core member against the retained
    # public metadata. Private Data references retain their reviewed bodies.
    surface = set(core['public_surface'])
    public_members = {line.split(' ', 3)[1] + '::' + line.split(' ', 3)[3]
                      for line in surface if line.startswith('MEMBER ') and
                      ' ATTRIBUTE ' not in line and ' FLAGS ' not in line and ' PARAMETER ' not in line}
    core_types = {line.split(' ')[1] for line in surface if line.startswith('TYPE ')}
    references = set()
    for body in observed_data['candidate_methods'].values():
        for instruction in json.loads(body)['instructions']:
            operand = instruction['operand']
            if isinstance(operand, str) and '::' in operand and operand.split('::')[0] in core_types:
                assert operand in public_members, operand
                references.add(operand)
    assert len(references) >= 20
    return dict(core_methods=3179, unchanged_core_methods=3178, data_methods=697,
        observer_original_methods_unchanged=696, observer_core_members=sorted(references),
        original_padcore_exact=True, candidate_build_flags_equal=True,
        selected_runtime_flags_pending=True, public_surface_equal=True)


def review():
    inputs = {}
    def retain(path):
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
        return read(path)
    for folder, digest in [(QUALIFIED, '62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'),
                           (CANDIDATE, '7fbee9acdcedfa3e86ad32b573c36a5bbd493be050d798509b0afd1acc80c9db'),
                           (SCREEN, '8e757188ca7a29c9c78a9fdc1803eb64d16dab18c476e0e433419eafd5245d73')]:
        closure = retain(folder / 'closed.json')
        assert pin(folder / 'closed.json')['sha256'] == digest and closure['passed']
        for name in ['analysis.json', 'collected/inventory/instructions.json']:
            if name in closure['files']:
                assert pin(folder / name) == closure['files'][name]
    assert not read(SCREEN / 'closed.json')['admitted']
    observer_review = retain(OBSERVER / 'build-review.json')
    assert observer_review['passed']
    assert observer_review['inventory']['sha256'] == '6b0a9df024d2b016be7655b1b4ca20a426d0e3132ed7ae194a3fe197fc491e78'
    assert pin(OBSERVER / 'build-collected/inventory/instructions.json') == observer_review['inventory']
    root = retain(QUALIFIED / 'collected/inventory/instructions.json')
    candidate = retain(CANDIDATE / 'collected/inventory/instructions.json')
    observer = retain(OBSERVER / 'build-collected/inventory/instructions.json')
    checked = reconcile(root, candidate, observer)
    products = retain(SCREEN / 'analysis.json')['products']
    for role, folder in [('current', SELECTED / 'collected/runtimes/current'),
                         ('candidate', CANDIDATE / 'collected/runtime')]:
        for name, wanted in products[role].items():
            assert pin(folder / name) == wanted
            inputs[(folder / name).relative_to(ROOT).as_posix()] = wanted
    for name, row in zip(['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], root['observations'], strict=True):
        assert row['before_sha256'] == products['current'][name]['sha256']
    for name, row in zip(['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll'], candidate['observations'], strict=True):
        assert row['after_sha256'] == products['candidate'][name]['sha256']
    for path, digest in [(OBSERVER / 'build-collected/runtime-observed/Lokad.Onnx.Data.dll', DATA_SHA),
                         (CONSUMER / 'build-collected/runtime-control/SampledAudio.dll', RUNNER_SHA)]:
        assert pin(path)['sha256'] == digest
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    return dict(passed=True, diagnostic_only=True, rejected_screen_preserved=True,
        products=products, observer=pin(OBSERVER / 'build-collected/runtime-observed/Lokad.Onnx.Data.dll'),
        consumer=pin(CONSUMER / 'build-collected/runtime-control/SampledAudio.dll'),
        compiled=checked, inputs=inputs)


if __name__ == '__main__':
    value = review()
    print(json.dumps({k:v for k,v in value.items() if k != 'inputs'}, indent=2))

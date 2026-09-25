"""Bind the measured source and portable tests to the actual release inventory."""
import importlib.util
from pathlib import Path
from protocol import pin, read

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
SOURCE = ROOT/'artifacts/parakeet-owned-batch-isolation-source-20260925'
SELECTED = ROOT/'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924'
REVIEW = ROOT/'artifacts/parakeet-owned-batch-isolation-root-scope-20260925'
OWNED_PUBLIC = ROOT/'artifacts/parakeet-packed-final-row-public-tests-20260925'
DEPTHWISE_PUBLIC = ROOT/'artifacts/parakeet-direct-depthwise-public-tests-20260925'
OWNED_TOOLS = TOOLS.parent/'packed-final-row-root-amd'
DEPTHWISE_TOOLS = TOOLS.parent/'direct-depthwise-public-tests'
TEST = 'tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
DEPTHWISE = 'tests/Lokad.Onnx.Backend.Tests/DirectDepthwiseTests.cs'
TEMPLATES = {TEST: OWNED_TOOLS/'OwnedPackedWeightTests.cs.txt', DEPTHWISE: DEPTHWISE_TOOLS/'DirectDepthwiseTests.cs.txt'}
CHANGED = sorted([
    'src/Lokad.Onnx.Data/ParakeetTranscriber.cs',
    'src/Lokad.Onnx/CPUExecutionProvider.MatMul.cs',
    'src/Lokad.Onnx/ComputationalGraph.cs', 'src/Lokad.Onnx/Global.cs',
    'src/Lokad.Onnx/GraphOwnedPacking.cs', 'src/Lokad.Onnx/OwnedPackedTensor.cs',
    'src/Lokad.Onnx/PackedFinalRowKernel.cs', 'src/Lokad.Onnx/TensorAlias.cs',
    'src/Lokad.Onnx/TensorOps.ConvPool.cs', 'src/Lokad.Onnx/TensorOps.MatMul.cs',
    'src/Lokad.Onnx/TensorOps.OwnedPackedMatMul.cs', 'src/Lokad.Onnx/TensorSlice.cs',
    'src/Lokad.Onnx/Zzz.DirectDepthwise.cs', 'src/Lokad.Onnx/Zzz.WideProjectionEntry.cs',
    TEST, DEPTHWISE, 'tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'])
ADDED = sorted(['src/Lokad.Onnx/GraphOwnedPacking.cs', 'src/Lokad.Onnx/OwnedPackedTensor.cs',
    'src/Lokad.Onnx/PackedFinalRowKernel.cs', 'src/Lokad.Onnx/TensorOps.OwnedPackedMatMul.cs',
    'src/Lokad.Onnx/Zzz.DirectDepthwise.cs', TEST, DEPTHWISE,
    'tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'])
RECEIPT = '961d4224c029c3c78d9148b0134c1585b5e87709e1d3eb013090286d585f2c9e'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def delta(source):
    before, after = source['before'], source['source']
    assert len(before) == 427 and len(after) == 435
    assert all(name not in before for name in ADDED)
    assert set(after) == set(before) | set(ADDED)
    changed = sorted(name for name, wanted in after.items() if before.get(name) != wanted)
    assert changed == CHANGED
    return changed


def verify_templates():
    owned = read(OWNED_PUBLIC/'prepared.json')
    assert owned['passed'] and owned['source'] == pin(SOURCE/'source'/TEST)
    assert owned['output'] == pin(TEMPLATES[TEST]) and owned['generator'] == pin(OWNED_TOOLS/'public_tests.py')
    verifier = load('retained_owned_public_verifier', OWNED_TOOLS/'public_tests.py')
    assert owned['new_backend_cases'] == verifier.qualified_census()
    assert owned['review'] == verifier.verify((SOURCE/'source'/TEST).read_text(encoding='utf8'), TEMPLATES[TEST].read_text(encoding='utf8'))
    depthwise = read(DEPTHWISE_PUBLIC/'prepared.json')
    assert depthwise['passed'] and depthwise['prepared_source_only']
    assert depthwise['source'] == pin(SOURCE/'source'/DEPTHWISE)
    assert depthwise['output'] == pin(TEMPLATES[DEPTHWISE]) and depthwise['preparer'] == pin(DEPTHWISE_TOOLS/'prepare.py')
    verifier = load('retained_depthwise_public_verifier', DEPTHWISE_TOOLS/'prepare.py')
    assert verifier.portable((SOURCE/'source'/DEPTHWISE).read_text(encoding='utf8'), depthwise['geometries']) == TEMPLATES[DEPTHWISE].read_text(encoding='utf8')
    return owned, depthwise


def verify_source():
    assert pin(SOURCE/'prepared.json')['sha256'] == RECEIPT
    prepared = read(SOURCE/'prepared.json')
    assert prepared['passed'] and prepared['source_reversible'] and prepared['shared_dispatcher_matches_release']
    assert pin(REVIEW/'review.json')['sha256'] == '1efdc618bfe795f31e577b8fed56a08892a1a9bcfbcfc0d317da47f48b7a4aaa'
    review = read(REVIEW/'review.json')
    assert review['passed'] and review['source_receipt'] == pin(SOURCE/'prepared.json')
    assert review['patch'] == pin(REVIEW/'integration.patch')
    baseline = {name.removeprefix('source/'): wanted for name, wanted in read(SELECTED/'bundle/stage.json')['files'].items() if name.startswith('source/')}
    source = dict(before=baseline, source=prepared['source'])
    assert review['baseline_files'] == baseline and review['measured_files'] == prepared['source']
    assert review['changed'] == delta(source) and review['added'] == ADDED
    for name, wanted in source['source'].items():
        path = SOURCE/'source'/name
        assert path.resolve().is_relative_to((SOURCE/'source').resolve()) and pin(path) == wanted, name
    assert review['portable_owned'] == pin(OWNED_PUBLIC/'prepared.json')
    assert review['portable_depthwise'] == pin(DEPTHWISE_PUBLIC/'prepared.json')
    verify_templates()
    assert root_files(source) == review['intended_root_files']
    return source


def root_files(source):
    delta(source)
    result = dict(source['source'])
    result.update({name: pin(path) for name, path in TEMPLATES.items()})
    assert len(result) == 435
    return result


def verify_before(source):
    delta(source)
    for name, wanted in source['before'].items():
        path = ROOT/name
        assert path.resolve().is_relative_to(ROOT) and pin(path) == wanted, name
    assert all(not (ROOT/name).exists() for name in ADDED)


if __name__ == '__main__':
    source = verify_source()
    verify_before(source)
    print(dict(passed=True, before=427, measured=435, root=435, changed=delta(source), root_product_changed=False))

"""Bind the actual release-to-M78 source delta and qualified portable tests."""
from pathlib import Path
from protocol import pin,read
from public_tests import verify as verify_public,qualified_census

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
SELECTED=ROOT/'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924'
PUBLIC=ROOT/'artifacts/parakeet-packed-final-row-public-tests-20260925'
TEST='tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
CHANGED=sorted([
    'src/Lokad.Onnx.Data/ParakeetTranscriber.cs',
    'src/Lokad.Onnx/CPUExecutionProvider.MatMul.cs',
    'src/Lokad.Onnx/ComputationalGraph.cs',
    'src/Lokad.Onnx/Global.cs',
    'src/Lokad.Onnx/GraphOwnedPacking.cs',
    'src/Lokad.Onnx/OwnedPackedTensor.cs',
    'src/Lokad.Onnx/PackedFinalRowKernel.cs',
    'src/Lokad.Onnx/TensorAlias.cs',
    'src/Lokad.Onnx/TensorOps.MatMul.cs',
    'src/Lokad.Onnx/TensorOps.OwnedPackedMatMul.cs',
    'src/Lokad.Onnx/TensorSlice.cs',
    'src/Lokad.Onnx/Zzz.WideProjectionEntry.cs',
    TEST,'tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'])
ADDED=sorted(['src/Lokad.Onnx/GraphOwnedPacking.cs','src/Lokad.Onnx/OwnedPackedTensor.cs',
    'src/Lokad.Onnx/PackedFinalRowKernel.cs','src/Lokad.Onnx/TensorOps.OwnedPackedMatMul.cs',
    TEST,'tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'])
RECEIPT='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'


def delta(source):
    before,after=source['before'],source['source']
    assert len(before)==427 and len(after)==433
    assert all(name not in before for name in ADDED)
    assert set(after)==set(before)|set(ADDED)
    changed=sorted(name for name,wanted in after.items() if before.get(name)!=wanted)
    assert changed==CHANGED
    return changed


def verify_source():
    assert pin(SOURCE/'prepared.json')['sha256']==RECEIPT
    prepared=read(SOURCE/'prepared.json');assert prepared['passed']
    baseline={n.removeprefix('source/'):v for n,v in read(SELECTED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    # The isolated preparation starts at M76. Integration starts at the actual
    # qualified release, with its own retained 427-file source inventory.
    source={**prepared,'before':baseline,'isolated_before':prepared['before']}
    delta(source)
    for name,wanted in source['source'].items():
        path=SOURCE/'source'/name;assert path.resolve().is_relative_to((SOURCE/'source').resolve())
        assert pin(path)==wanted,name
    receipt=read(PUBLIC/'prepared.json');assert receipt['passed']
    assert receipt['prepared']==pin(SOURCE/'prepared.json')
    assert receipt['source']==pin(SOURCE/'source'/TEST)
    assert receipt['output']==pin(TOOLS/'OwnedPackedWeightTests.cs.txt')
    assert receipt['generator']==pin(TOOLS/'public_tests.py')
    assert receipt['new_backend_cases']==qualified_census()
    assert receipt['review']==verify_public((SOURCE/'source'/TEST).read_text(encoding='utf8'),
                                           (TOOLS/'OwnedPackedWeightTests.cs.txt').read_text(encoding='utf8'))
    return source


def root_files(source):
    delta(source);result=dict(source['source'])
    result[TEST]=pin(TOOLS/'OwnedPackedWeightTests.cs.txt')
    assert len(result)==433
    return result


def verify_before(source):
    delta(source)
    for name,wanted in source['before'].items():
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT)
        assert pin(path)==wanted,name
    assert all(not (ROOT/name).exists() for name in ADDED)


if __name__=='__main__':
    source=verify_source();verify_before(source)
    print(dict(passed=True,before=427,measured=433,root=433,changed=delta(source),root_product_changed=False))

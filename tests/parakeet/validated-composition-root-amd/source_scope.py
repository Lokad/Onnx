"""Verify the measured source and exact root delta without modifying product files."""
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924'
SELECTED=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
CHANGED=['src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs', 'src/Lokad.Onnx/ComputationalGraph.cs', 'src/Lokad.Onnx/GraphExecution.cs', 'src/Lokad.Onnx/GraphLstmPacking.cs', 'src/Lokad.Onnx/GraphPacking.cs', 'src/Lokad.Onnx/TensorExecutionOptions.cs', 'src/Lokad.Onnx/TensorSlice.cs', 'tests/Lokad.Onnx.Backend.Tests/PreparedLstmWeightsTests.cs', 'tests/Lokad.Onnx.Tensors.Tests/SliceReshapeCopyTests.cs']
ADDED=['src/Lokad.Onnx/GraphLstmPacking.cs', 'tests/Lokad.Onnx.Backend.Tests/PreparedLstmWeightsTests.cs', 'tests/Lokad.Onnx.Tensors.Tests/SliceReshapeCopyTests.cs']
RECEIPT='af68e6c2c28f5794f3eece39bd9e78fc6809e2ca8925a5616ff2956c405061c4'


def delta(source):
    before=source['before'];after=source['source']
    assert len(before)==422 and len(after)==425
    assert all(name not in before for name in ADDED)
    assert set(after)==set(before)|set(ADDED)
    changed=sorted(name for name,wanted in after.items() if before.get(name)!=wanted)
    assert changed==CHANGED
    return changed


def verify_source():
    assert pin(SOURCE/'prepared.json')['sha256']==RECEIPT
    source=read(SOURCE/'prepared.json');assert source['passed']
    baseline={n.removeprefix('source/'):v for n,v in read(SELECTED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert source['before']==baseline
    delta(source)
    for name,wanted in source['source'].items():
        path=SOURCE/'source'/name
        assert path.resolve().is_relative_to((SOURCE/'source').resolve())
        assert pin(path)==wanted,name
    return source


def verify_before(source):
    delta(source)
    for name,wanted in source['before'].items():
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT)
        assert pin(path)==wanted,name
    for name in ADDED:assert not (ROOT/name).exists(),name


if __name__=='__main__':
    source=verify_source();verify_before(source)
    print(dict(passed=True,before=len(source['before']),source=len(source['source']),
               changed=delta(source),root_product_changed=False))

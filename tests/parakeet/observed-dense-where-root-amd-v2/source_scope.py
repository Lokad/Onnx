"""Verify the measured two-file product change before adding public regression tests."""
from pathlib import Path
from protocol import pin,read
from nullability import correct,bytes_pin,verify_failure,HELPER

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
SELECTED=ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
CHANGED=['src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs','src/Lokad.Onnx/Zzz.DenseScalarWhere.cs']
ADDED=['src/Lokad.Onnx/Zzz.DenseScalarWhere.cs']
TEST='tests/Lokad.Onnx.Backend.Tests/DenseScalarWhereTests.cs'
RECEIPT='41f2477c7060c3f543471bebe991c77ea3b853d724708fec098a2cbb505cb0a2'


def delta(source):
    before=source['before'];after=source['source']
    assert len(before)==425 and len(after)==426
    assert all(name not in before for name in ADDED)
    assert set(after)==set(before)|set(ADDED)
    changed=sorted(name for name,wanted in after.items() if before.get(name)!=wanted)
    assert changed==CHANGED
    return changed


def verify_source():
    failure=verify_failure()
    assert pin(SOURCE/'prepared.json')['sha256']==RECEIPT
    source=read(SOURCE/'prepared.json');assert source['passed']
    baseline={n.removeprefix('source/'):v for n,v in read(SELECTED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert source['before']==baseline
    delta(source)
    for name,wanted in source['source'].items():
        path=SOURCE/'source'/name
        assert path.resolve().is_relative_to((SOURCE/'source').resolve())
        assert pin(path)==wanted,name
    assert TEST not in source['source']
    assert source['source'][HELPER]==failure['helper']
    return source


def root_files(source):
    delta(source)
    result=dict(source['source']);result[TEST]=pin(TOOLS/'DenseScalarWhereTests.cs.txt')
    result[HELPER]=bytes_pin(correct((SOURCE/'source'/HELPER).read_bytes()))
    assert len(result)==427
    return result


def verify_before(source):
    delta(source)
    for name,wanted in source['before'].items():
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT)
        assert pin(path)==wanted,name
    for name in [*ADDED,TEST]:assert not (ROOT/name).exists(),name


if __name__=='__main__':
    source=verify_source()
    for name,wanted in root_files(source).items():assert pin(ROOT/name)==wanted,name
    print(dict(passed=True,before=len(source['before']),measured=len(source['source']),
               root=len(root_files(source)),changed=delta(source),nullable_contract_corrected=True))

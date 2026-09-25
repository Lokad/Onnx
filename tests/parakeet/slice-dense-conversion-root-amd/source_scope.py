"""Integrate only the measured override and the corrected, already qualified tests."""
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
SOURCE=ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924'
SELECTED=ROOT/'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924'
TESTS=ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924'
TEST='tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'
CHANGED=['src/Lokad.Onnx/TensorSlice.cs',TEST]
ADDED=[TEST]
RECEIPT='a9811c6369122eb8b53ab1d88d8329928a999d172d89786a1b393c3451e91e1e'


def delta(source):
    before,after=source['before'],source['source']
    assert len(before)==427 and len(after)==428
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
        path=SOURCE/'source'/name;assert path.resolve().is_relative_to((SOURCE/'source').resolve())
        assert pin(path)==wanted,name
    review=read(TESTS/'build-review.json')
    assert pin(TESTS/'build-review.json')['sha256']=='0eeef7982a09859c8f791bc361d900ab04941ada148e5600527978732b5351d4'
    assert review['passed'] and not review['product_rebuilt']
    correction=review['corrected_test'];assert correction['file']=='source/'+TEST
    original=(SOURCE/'source'/TEST).read_bytes()
    before,after=correction['before'].encode(),correction['after'].encode()
    assert original.count(before)==1
    expected=original.replace(before,after)
    qualified=TESTS/'bundle/source'/TEST
    assert qualified.read_bytes()==expected==(TOOLS/'SliceDenseConversionTests.cs.txt').read_bytes()
    assert pin(qualified)==read(TESTS/'bundle/spec.json')['files']['source/'+TEST]
    return source


def root_files(source):
    delta(source);result=dict(source['source'])
    result[TEST]=pin(TOOLS/'SliceDenseConversionTests.cs.txt')
    assert len(result)==428
    return result


def verify_before(source):
    delta(source)
    for name,wanted in source['before'].items():
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT)
        assert pin(path)==wanted,name
    assert not (ROOT/TEST).exists()


if __name__=='__main__':
    source=verify_source();verify_before(source)
    print(dict(passed=True,before=427,measured=428,root=428,changed=delta(source),root_product_changed=False))

"""Verify the measured source and exact root delta without modifying product files."""
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
SELECTED=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
CHANGED=['src/Lokad.Onnx/TensorOps.MatMul.cs',
         'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs',
         'src/Lokad.Onnx/Zzz.WideProjectionEntry.cs']
ADDED=CHANGED[1:]
RECEIPT='72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'


def delta(source):
    before=source['before'];after=source['source']
    assert len(before)==420 and len(after)==422
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

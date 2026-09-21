"""Close the local candidate proof, including the preserved root-discovery failure."""
import json
from pathlib import Path
from prepare import ROOT, BASE, TOOLS, pin, save
from run import absent


def main():
    target=BASE/'qualification-closed.json';assert not target.exists()
    comparison=json.loads((BASE/'closed.json').read_text())
    assert comparison['passed']
    for name,expected in comparison['files'].items():assert pin(ROOT/name)==expected,name
    old=json.loads((BASE/'qualification.json').read_text());new=json.loads((BASE/'qualification-v2.json').read_text())
    audit=json.loads((BASE/'qualification-audit.json').read_text())
    assert old['complete'] and old['code']==1 and old['models']==[]
    assert new['complete'] and new['code']==0 and audit['passed']
    assert audit['qualification']==pin(BASE/'qualification-v2.json')
    identities=comparison['identities']+[old['supervisor']]+audit['identities']
    assert len(identities)==9 and all(absent(p) for p in identities)
    logs=BASE/'logs'
    assert 'Failed:     0, Passed:  3115, Skipped:    93' in (logs/'root-corrected-backend-full-tests.log').read_text()
    assert 'Failed:     0, Passed:   342, Skipped:     0' in (logs/'root-corrected-tensors-full-tests.log').read_text()
    spec=json.loads((BASE/'manifest.json').read_text())
    assert pin(BASE/'candidate-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll')==spec['cores']['candidate']
    # Confirm the archived product source differs only in the declared two files.
    left=BASE/'baseline-source/src';right=BASE/'candidate-source/src'
    changes=[]
    for path in left.rglob('*'):
        if not path.is_file() or {'bin','obj'}.intersection(path.relative_to(left).parts):continue
        other=right/path.relative_to(left)
        if pin(path)!=pin(other):changes.append(path.relative_to(left).as_posix())
    assert sorted(changes)==['Lokad.Onnx/MathOps.cs','Lokad.Onnx/TensorOps.ConvPool.cs']
    files=dict(comparison['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in (BASE/'shared-runtime').iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    save(target,dict(passed=True,files=files,identities=identities,source_changes=changes,comparison=pin(BASE/'closed.json'),
        qualification_audit=pin(BASE/'qualification-audit.json'),backend_passed=3115,backend_skipped=93,tensors_passed=342))
    print(json.dumps(dict(passed=True,files=len(files),identities=len(identities),closure=pin(target))))


if __name__=='__main__':main()

"""Bind the complete successor comparison, source changes and model qualification."""
import json
from prepare import ROOT, BASE, OLD, TOOLS, pin, save
from run import absent


def main():
    target=BASE/'qualification-closed.json';assert not target.exists()
    comparison=json.loads((BASE/'closed.json').read_text());assert comparison['passed']
    for name,expected in comparison['files'].items():assert pin(ROOT/name)==expected,name
    analysis=json.loads((BASE/'analysis.json').read_text())
    assert analysis['passed'] and analysis['output_arrays']==72 and analysis['public_calls']==64
    assert all(c.get('bit_identical',True) for c in analysis['comparisons'])
    state=json.loads((BASE/'qualification.json').read_text());audit=json.loads((BASE/'qualification-audit.json').read_text())
    assert state['complete'] and state['code']==0 and audit['passed']
    assert audit['qualification']==pin(BASE/'qualification.json')
    assert audit['arrays']==166 and audit['values']==5000814
    identities=comparison['identities']+audit['identities']
    assert len(identities)==8 and all(absent(p) for p in identities)
    logs=BASE/'logs'
    expected={'focused-tests':(141,0),'focused-fallback-tests':(28,0),
        'backend-full-tests':(3129,93),'tensors-full-tests':(342,0)}
    for name,(passed,skipped) in expected.items():
        log=(logs/(name+'.log')).read_text()
        assert f'Failed:     0, Passed: {passed:5d}, Skipped: {skipped:5d}' in log,(name,log)
    spec=json.loads((BASE/'manifest.json').read_text())
    assert pin(BASE/'candidate-source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll')==spec['cores']['candidate']
    left=OLD/'candidate-source/src';right=BASE/'candidate-source/src'
    def source_files(base):return {p.relative_to(base).as_posix():p for p in base.rglob('*')
        if p.is_file() and not {'bin','obj'}.intersection(p.relative_to(base).parts)}
    a=source_files(left);b=source_files(right);assert a.keys()==b.keys()
    changes=sorted(name for name in a if pin(a[name])!=pin(b[name]))
    assert changes==['Lokad.Onnx/MathOps.cs','Lokad.Onnx/TensorOps.ConvPool.cs']
    files=dict(comparison['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    save(target,dict(passed=True,files=files,identities=identities,source_changes=changes,
        comparison=pin(BASE/'closed.json'),qualification_audit=pin(BASE/'qualification-audit.json'),tests=expected))
    print(json.dumps(dict(passed=True,files=len(files),identities=len(identities),closure=pin(target))))


if __name__=='__main__':main()

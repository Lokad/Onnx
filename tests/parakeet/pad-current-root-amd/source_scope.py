"""Bind the 435 qualified inputs plus the measured Pad code and six explicit-argument facts."""
import importlib.util
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'owned-batch-isolation-root-policy-amd'
SOURCE=ROOT/'artifacts/parakeet-pad-current-source-20260926'
FIXTURE=ROOT/'artifacts/parakeet-pad-integration-tests-20260926'
QUALIFIED=ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
APPLIED=ROOT/'artifacts/parakeet-pad-current-root-integration-20260926'
SHAPE='src/Lokad.Onnx/CPUExecutionProvider.Shape.cs'
HELPER='src/Lokad.Onnx/Zzz.LastAxisPadDispatch.cs'
TEST='tests/Lokad.Onnx.Backend.Tests/LastAxisPadTests.cs'
CHANGED=[SHAPE,HELPER,TEST]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def corrected_fixture(before):
    edits=[(b'T fill, bool reversed = false)',b'T fill, bool reversed)'),
           (b'Check(logical, shape, pads, fill);',b'Check(logical, shape, pads, fill, false);'),
           (b'Check(Array.Empty<T>(), new[] { 2, 0, 3 }, new[] { 0, 0, 4, 0, 0, 4 }, fill);',
            b'Check(Array.Empty<T>(), new[] { 2, 0, 3 }, new[] { 0, 0, 4, 0, 0, 4 }, fill, false);'),
           (b'Check(data, new[] { 2, 3, 4 }, new[] { 1, -1, -1, 0, 1, 2 }, fill);',
            b'Check(data, new[] { 2, 3, 4 }, new[] { 1, -1, -1, 0, 1, 2 }, fill, false);')]
    after=before
    for old,new in edits:
        assert after.count(old)==1;after=after.replace(old,new)
    assert after.count(b'[Fact]')==6 and b'bool reversed = false' not in after
    return after


def verify_source():
    assert pin(SOURCE/'prepared.json')['sha256']=='f9eb6c4a26362529d4fe19089e35201d5e26f318561445b58078d05c1df5dbea'
    source=read(SOURCE/'prepared.json');assert source['passed'] and source['changed']==CHANGED
    assert source['qualified_parent']==pin(QUALIFIED/'closed.json')
    assert source['qualified_parent']['sha256']=='c77ef606c76508144c5656bdbe48aeac8948ce28396c8bee645587d31609c475'
    qualified=read(QUALIFIED/'closed.json');assert qualified['passed']
    prior=QUALIFIED/'bundle/evidence/root-applied.json'
    assert qualified['files']['bundle/evidence/root-applied.json']==pin(prior)==source['root_integration']
    assert read(prior)['source_files']==source['before'] and len(source['before'])==435
    assert set(source['source'])==set(source['before'])|{HELPER,TEST}
    assert [n for n,v in source['source'].items() if source['before'].get(n)!=v]==CHANGED
    assert len(source['source'])==437
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    fixture=read(FIXTURE/'prepared.json')
    assert fixture['passed'] and fixture['source_prepared']==pin(SOURCE/'prepared.json')
    assert fixture['assertions_and_test_names_unchanged'] and fixture['product_unchanged']
    assert fixture['original']==source['source'][TEST] and fixture['corrected']==pin(FIXTURE/'LastAxisPadTests.cs')
    assert fixture['patch']==pin(FIXTURE/'review.patch')
    assert fixture['script']==pin(TOOLS.parent/'pad-current-integration/prepare_tests.py')
    assert (FIXTURE/'LastAxisPadTests.cs').read_bytes()==corrected_fixture((SOURCE/'source'/TEST).read_bytes())
    guard='tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs'
    assert fixture['source_policy_test']==source['before'][guard]==source['source'][guard]
    return source


def root_files(source):
    result=dict(source['source']);result[TEST]=pin(FIXTURE/'LastAxisPadTests.cs')
    assert len(result)==437
    return result


def verify_root(files):
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    prefixes=['src/','tests/Lokad.Onnx.Backend.Tests/','tests/Lokad.Onnx.Tensors.Tests/']
    actual={p.relative_to(ROOT).as_posix() for prefix in prefixes for p in (ROOT/prefix).rglob('*')
            if p.is_file() and not {'bin','obj'}&set(p.relative_to(ROOT).parts)}
    assert actual=={n for n in files if any(n.startswith(prefix) for prefix in prefixes)}
    return True


if __name__=='__main__':
    source=verify_source();print(dict(passed=True,files=len(root_files(source)),changed=CHANGED,root_applied=False))

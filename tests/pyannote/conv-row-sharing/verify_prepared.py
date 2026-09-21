"""Close local preparation while explicitly leaving AMD execution unqualified."""
import json
from pathlib import Path
import re
from prepare import ROOT, TOOLS, OLD, pin, save

BASE=ROOT/'artifacts/pyannote-conv-row-sharing-v2-20260921'


def main():
    target=BASE/'local-preparation.json';assert not target.exists()
    prepared=json.loads((BASE/'prepared.json').read_text())
    assert pin(BASE/'prepared.json')['sha256']=='724630645d35446464ce298c182b3bd7f2ff79d9e6d01e50a4ebe540bde09f1a'
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    failure=json.loads((BASE/'predecessor-failure.json').read_text());path=ROOT/failure['path']
    assert pin(path)=={k:failure[k] for k in ['bytes','sha256']}
    failed=json.loads(path.read_text());assert not failed['passed'] and failed['product_compiled'] and not failed['tests_executed']
    for name,wanted in failed['files'].items():assert pin(ROOT/name)==wanted,name
    old_test=path.parent/'candidate-source/tests/Lokad.Onnx.Backend.Tests/ConvPackedRowsTests.cs'
    new_test=BASE/'candidate-source/tests/Lokad.Onnx.Backend.Tests/ConvPackedRowsTests.cs'
    assert old_test.read_text().replace('result.Dims.ToArray()','result.Dimensions.ToArray()')==new_test.read_text()
    for name in ['TensorOps.ConvPool.cs','TensorOps.ConvPackedRows.cs']:
        assert pin(path.parent/'candidate-source/src/Lokad.Onnx'/name)==pin(BASE/'candidate-source/src/Lokad.Onnx'/name)
    builds=json.loads((BASE/'builds.json').read_text());assert all(r['code']==0 for r in builds) and len(builds)==3
    build=(BASE/'logs/backend-build.log').read_text();assert '0 Warning(s)' in build and '0 Error(s)' in build
    tests=[]
    for name,passed,skipped in [('focused-tests',148,3),('focused-fallback-tests',37,2)]:
        log=(BASE/'logs'/(name+'.log')).read_text()
        match=re.search(r'Failed:\s*(\d+), Passed:\s*(\d+), Skipped:\s*(\d+), Total:\s*(\d+)',log)
        assert match and tuple(map(int,match.groups()))==(0,passed,skipped,passed+skipped)
        tests.append(dict(name=name,passed=passed,skipped=skipped))
    changed=[]
    source=BASE/'candidate-source'
    for p in (source/'src').rglob('*'):
        if not p.is_file() or {'bin','obj'}.intersection(p.relative_to(source).parts):continue
        original=OLD/'candidate-source'/p.relative_to(source)
        if not original.exists() or pin(original)!=pin(p):changed.append(str(p.relative_to(source)).replace('\\','/'))
    assert set(changed)==set(prepared['changed_source'])=={'src/Lokad.Onnx/TensorOps.ConvPool.cs','src/Lokad.Onnx/TensorOps.ConvPackedRows.cs'}
    assert pin(BASE/'runtime/Lokad.Onnx.dll')==prepared['core'] and pin(BASE/'runtime/Lokad.Onnx.Data.dll')==prepared['data']
    result=dict(local_preparation_passed=True,avx512_execution_qualified=False,full_model_qualified=False,performance_qualified=False,
        prepared=pin(BASE/'prepared.json'),initial_failure=failure,tests=tests,changed_source=changed,core=prepared['core'],
        verified_files=len(prepared['files']),verifier=pin(Path(__file__)),scope=prepared['scope'])
    save(target,result);print(json.dumps(result))


if __name__=='__main__':main()

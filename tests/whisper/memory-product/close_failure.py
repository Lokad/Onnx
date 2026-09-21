"""Preserve the completed coherent-build source-policy failure without replay."""
import xml.etree.ElementTree as ET
from common import *


def main():
    base=ROOT/'artifacts/whisper-memory-product-20260921';state=read(base/'run.json');prepared=read(base/'prepared.json')
    assert state['complete'] and state['code']==1 and [r['name'] for r in state['runs']]==['solution-build','tensor-tests']
    assert [r['code'] for r in state['runs']]==[0,1] and state['prepared']==pin(base/'prepared.json')
    births=[state['supervisor']];samples=0
    for run in state['runs']:
        assert run['complete'] and 0<run['seconds']<300
        births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
        rows=[json.loads(s) for s in (base/(run['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(rows)==run['samples']
        for row in rows:
            assert row['available']>=1024**3 and sum(m['rss'] for m in row['members'])<4*1024**3
            assert all(m['affinity']==[0] and m['birth']==run['members'][str(m['pid'])] for m in row['members'])
        samples+=len(rows)
    assert all(absent(b) for b in births)
    for name,wanted in read(base/'source-manifest.json').items():assert pin(base/'source'/name)==wanted,name
    for name,wanted in prepared['tools'].items():assert pin(ROOT/name)==wanted,name
    tests=ET.parse(base/'test-results/tensors.trx').getroot();counters=next(e for e in tests.iter() if e.tag.endswith('Counters'))
    assert counters.attrib['passed']=='340' and counters.attrib['failed']=='2' and counters.attrib['total']=='342'
    failures=[e.attrib['testName'] for e in tests.iter() if e.tag.endswith('UnitTestResult') and e.attrib.get('outcome')=='Failed']
    assert set(failures)=={'Lokad.Onnx.Tensors.Tests.NoOptionalParametersTests.SourceTree_HasNoOptionalParameters','Lokad.Onnx.Tensors.Tests.NoNullForgivingTests.SourceTree_HasNoNullForgivingOperators'}
    report=Path(__file__).parent/'failure-20260921.md'
    report.write_text('''# Private memory candidate: coherent source-policy failure

The complete eight-project Release solution builds successfully, with four
nullable warnings in the new sharing tests. The tensor suite passes 340 tests
and fails two repository source-policy checks: four null-forgiving operators in
the private Data sharing helper and two optional parameters in its test helper.
No package or package consumer was executed after this failure.

The earlier 3,101-test backend pass did not exercise these tensor-suite source
policies. Its functional evidence and the separately running AMD inference remain
valid within their scopes; they do not make this coherent qualification pass.

Preserve the original build, all 342 test outcomes, source and resource samples
under `artifacts/whisper-memory-product-20260921`. All owned process identities
are terminal. A distinct candidate will replace null-forgiving syntax with an
accurate nullable flow annotation and replace optional test parameters with
explicit overloads. Runtime behavior must be checked separately; no production
source, arithmetic tolerance or test exclusion changes in this failed attempt.
''',encoding='utf-8')
    write(base/'failure-closed.json',dict(closure_passed=True,campaign_passed=False,failures=failures,counters=counters.attrib,
        resource_samples=samples,births=births,report=pin(report),files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(closure_passed=True,campaign_passed=False,receipt=pin(base/'failure-closed.json'),samples=samples,births=len(births))))


if __name__=='__main__':main()

"""Preserve the complete root run stopped by the no-null-forgiving source guard."""
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/observed-dense-where-root-amd'))
from run import BASE,prepared
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from checks import census,inventory,suite

FAILED = 'Lokad.Onnx.Tensors.Tests.NoNullForgivingTests.SourceTree_HasNoNullForgivingOperators'
HELPER = 'src/Lokad.Onnx/Zzz.DenseScalarWhere.cs'


def main():
    assert not (BASE/'closed.json').exists()
    prepared()
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json')
    payload=read(BASE/'payload.json');transfer=read(BASE/'collection-transfer.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json') and transfer['passed']
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    assert state['complete'] and state['code']==1 and state['supervisor']==read(BASE/'deployment.json')
    assert state['boot_time']==1789634288.0 and state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==JOBS[:10]
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==(1 if row['name']=='tensors-tests' else 0)
        assert row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available']
        assert row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (c/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    for name,wanted in payload['files'].items():
        if name.startswith('source/'):
            assert pin(BASE/'bundle'/name)==pin(ROOT/name.removeprefix('source/'))==wanted,name
    built=read(c/'built.json');assert built['passed']
    for name,wanted in built['product'].items():assert pin(c/'runtime'/name)==wanted
    il=inventory(read(c/'inventory/instructions.json'),payload['measured'],built['product'])
    assert il==read(c/'inventory/review.json')
    backend=suite(c/'backend-tests/backend.trx','backend',c/'evidence')
    assert backend==read(c/'backend-tests/review.json') and (backend['passed'],backend['skipped'])==(3499,41)
    trx=c/'tensors-tests/tensors.trx';tree=ET.parse(trx)
    rows=tree.findall('.//{*}UnitTestResult');counts=tree.find('.//{*}Counters').attrib
    assert (len(rows),int(counts['total']),int(counts['passed']),int(counts['failed']))==(368,368,367,1)
    actual=Counter((r.attrib['testName'],r.attrib['outcome']) for r in rows)
    expected=census(c/'evidence/selected-tensors.trx')
    assert expected[(FAILED,'Passed')]==1
    expected[(FAILED,'Passed')]-=1;expected[(FAILED,'Failed')]+=1
    assert +expected==actual
    failure,=[r for r in rows if r.attrib['outcome']=='Failed']
    message=failure.find('.//{*}Message').text
    assert 'Null-forgiving operators found:' in message and HELPER in message and 'output = null!' in message
    assert (ROOT/HELPER).read_text().count('output = null!;')==1
    assert pin(ROOT/HELPER)==payload['files']['source/'+HELPER]
    analysis=dict(passed=False,release_admitted=False,classification='New helper violates the existing no-null-forgiving source policy.',
        product_tests_failed=1,failed_test=FAILED,failure_message=message,
        tensor_passed=367,tensor_failed=1,other_tensor_outcomes_exact=True,tensor_trx=pin(trx),
        helper=pin(ROOT/HELPER),policy_test=pin(ROOT/'tests/Lokad.Onnx.Tensors.Tests/NoNullForgivingTests.cs'),
        inventory=il,backend=backend,built=built['product'],measured=payload['measured'],resources=resources,
        completed_jobs=JOBS[:10],unexecuted_jobs=JOBS[10:],root_source_verified=True,
        normal_root_and_package_incomplete=True)
    save(BASE/'failure-analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=False,release_admitted=False,analysis=pin(BASE/'failure-analysis.json'),
        verifier=pin(__file__),files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    out=Path(__file__).parent/'root-source-policy-failure-20260924.json';assert not out.exists()
    save(out,dict(closure=pin(BASE/'closed.json'),**analysis))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),release_admitted=False,
        backend=backend,tensor_passed=367,tensor_failed=1,remaining_jobs=len(JOBS[10:]))))


if __name__=='__main__':main()

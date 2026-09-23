"""Preserve stopped V1 and prove the omitted hardware-only test census."""
import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-root-amd'))
from run import BASE,prepared
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from checks import census,inventory,suite,suite256


def main():
    assert not (BASE/'closed.json').exists()
    prepared();c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json')
    payload=read(BASE/'payload.json');transfer=read(BASE/'collection-transfer.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json') and transfer['passed']
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(c/'collection.json')
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    assert state['complete'] and state['code']==1 and state['supervisor']==read(BASE/'deployment.json')
    assert state['boot_time']==1789634288.0 and state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==JOBS[:11]
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert 'suite256' in state['error'] and 'Exp512Tests' in state['error']
    resources=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (c/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=gap<10 for gap in gaps)
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss']))
    built=read(c/'built.json');assert built['passed']
    for name,wanted in built['product'].items():assert pin(c/'runtime'/name)==wanted
    il=inventory(read(c/'inventory/instructions.json'),payload['measured'],built['product'])
    assert il==read(c/'inventory/review.json')
    ordinary={n:suite(c/(n+'-tests')/(n+'.trx'),n,c/'evidence') for n in ['backend','tensors']}
    for n,v in ordinary.items():assert v==read(c/(n+'-tests')/'review.json')
    original_failed=False
    try:suite256(c/'backend-tests-256/backend.trx','backend',c/'evidence')
    except AssertionError:original_failed=True
    assert original_failed
    path=ROOT/'tests/Lokad.Onnx.Backend.Tests/Exp512Tests.cs';source_pin=pin(path)
    assert source_pin['sha256']=='1cb029e34d24026bb532bac12b59d9d0534dca1a35f8664543a8f4dbeba7b36b'
    key='source/'+path.relative_to(ROOT).as_posix()
    assert source_pin==payload['files'][key]==read(ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923/bundle/stage.json')['files'][key]
    source=path.read_text()
    assert 'Vector512.IsHardwareAccelerated' in source and 'Fma.IsSupported' in source
    assert source.count('Skip.IfNot(ProbeAvailable(), "Requires AVX-512 plus FMA.");')==5
    assert source.count('[SkippableFact]')==4 and source.count('[SkippableTheory]')==1
    assert all(source.count('[InlineData('+str(n)+')]')==1 for n in [1,7,15,16,17,33])
    before=census(c/'evidence/selected-backend.trx');actual=census(c/'backend-tests-256/backend.trx')
    names=['Lokad.Onnx.Backend.Tests.Exp512Tests.'+n for n in ['ProbeHoldsContractOnWideSweep','ProbeHoldsContractOnReducedRange','ProbeMatchesEstrinCore','ProbeHandlesExceptionals']]
    names += ['Lokad.Onnx.Backend.Tests.Exp512Tests.ProbeTailsMatchScalar(n: '+str(n)+')' for n in [1,7,15,16,17,33]]
    for name in names:assert before[(name,'Passed')]==actual[(name,'NotExecuted')]==1
    spec=importlib.util.spec_from_file_location('corrected_root_checks',ROOT/'tests/parakeet/wide-entry-first-use-root-amd-v2/checks.py')
    corrected=importlib.util.module_from_spec(spec);spec.loader.exec_module(corrected)
    widths=corrected.suite256(c/'backend-tests-256/backend.trx','backend',c/'evidence')
    assert (widths['passed'],widths['skipped'])==(3359,131)
    analysis=dict(passed=False,classification='Checker omitted ten existing AVX512-only cases.',product_tests_failed=0,
        inventory=il,ordinary=ordinary,disabled_backend=widths,source=source_pin,omitted_cases=names,
        completed_jobs=JOBS[:11],unexecuted_jobs=JOBS[11:],resources=resources,
        release_admitted=False,normal_root_and_package_incomplete=True)
    save(BASE/'failure-analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=False,release_admitted=False,analysis=pin(BASE/'failure-analysis.json'),
        verifier=pin(__file__),files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    out=Path(__file__).parent/'root-census-correction-20260923.json'
    assert not out.exists();save(out,dict(closure=pin(BASE/'closed.json'),**analysis))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),release_admitted=False,omitted_cases=10,remaining_jobs=5)))


if __name__=='__main__':main()

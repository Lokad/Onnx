"""Independently inspect complete source, caller, suite and resource evidence."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import prepare as successor

c=SimpleNamespace(**successor.configured())


def resources(state_file,expected,inference):
    state=c.read(c.BASE / state_file);assert state['complete'] and state['code']==0
    assert [r['name'] for r in state['runs']]==expected
    identities=[state['supervisor']];result=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']['available']>=(10 if row['name'] in inference else 8)*1024**3
        assert row['preflight']==row['preflight_observations'][-1]
        assert all(r['seconds']<900 and r['disk']>=20*1024**3 for r in row['preflight_observations'])
        samples=[json.loads(s) for s in (c.BASE / 'logs' / (row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for s in samples:
            assert s['seconds']<900 and s['rss']<8*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
            assert s['rss']==sum(p['rss'] for p in s['members'])
            for p in s['members']:assert p['affinity']==[2] and row['members'][str(p['pid'])]==p['birth']
        identities.extend(dict(pid=int(pid),birth=birth) for pid,birth in row['members'].items())
        result.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    for identity in identities:c.terminal(identity)
    return dict(identities=identities,resources=result)


def preparation():
    proof=c.read(c.BASE / 'prepared.json');assert proof['passed'];c.verify(proof['files'])
    assert c.review()==c.read(c.BASE / 'instruction-review.json')
    assert not proof['production_changed'] and not proof['models_qualified'] and not proof['performance_qualified']
    schedule=[n+'-'+phase for n in ['cli','backend','tensors','bridge'] for phase in ['restore','build']]
    schedule+=['inventory','caller-restore','caller-build','caller-normal','caller-disabled']
    observations=resources('preparation.json',schedule,[])
    state=c.read(c.BASE / 'preparation.json')
    variants=[]
    for mode in ['normal','disabled']:
        result=c.read(c.BASE / ('caller-'+mode+'.json'))
        owner=next(r for r in state['runs'] if r['name']=='caller-'+mode)
        assert result['passed'] and result['pid']==owner['worker']['pid'] and result['processor_count']==1
        assert result['flags']==(['DOTNET_EnableHWIntrinsic'] if mode=='disabled' else [])
        assert result['fma']==(mode=='normal') and result['runtime']=='10.0.12'
        assert result['candidate']==proof['core']['sha256'] and result['baseline']==c.pin(c.CONTROL / 'Lokad.Onnx.dll')['sha256']
        rows=result['records'];assert len(rows)==400
        expected=[]
        for s in c.read(c.COMPONENT / 'shapes.json')['shapes']:
            expected.append((s['m'],s['n'],s['k'],1,True,'finite','Auto',0))
        for m in [32,64]:
            for n in [64,65]:
                for k in [1,2,7,8,31,32,33]:
                    for bias in [False,True]:
                        for pattern in ['finite','zero','special']:
                            expected.extend((m,n,k,2,bias,pattern,'Auto',p) for p in [0,1])
        for policy in ['Scalar','Simd']:
            for k in [2,8,33]:
                for bias in [False,True]:
                    for pattern in ['finite','zero','special']:expected.append((32,64,k,2,bias,pattern,policy,0))
        for m,n,k in [(5,9,2),(30,64,8),(33,64,2),(96,64,8),(32,63,32),(32,1024,65)]:
            expected.append((m,n,k,2,True,'finite','Auto',0))
        assert [tuple(r[k] for k in ['rows','reduction','block','groups','hasBias','pattern','policy','pass']) for r in rows]==expected
        for r in rows:
            assert r['checked_values']==2*r['rows']*r['groups']*(2*r['block']+1)+6
            assert len(r['digest'])==64 and 0<=r['nan_values']<=r['checked_values']
            if r['pattern']!='special':assert r['nan_values']==0
        payloads=0
        for r in rows:
            assert len(r['baseline_digest'])==64
            for d in r['nan_payload_differences']:
                left,right=int(d['baseline'],16),int(d['candidate'],16)
                assert left!=right and all((v & 0x7f800000)==0x7f800000 and (v & 0x7fffff)!=0 for v in [left,right])
                assert d['batch'] in [0,1] and 3<=d['index']<r['checked_values']-3
            payloads+=len(r['nan_payload_differences'])
            if r['pattern']!='special':
                assert r['baseline_digest']==r['digest'] and not r['nan_payload_differences']
        variants.append(dict(mode=mode,cases=len(rows),checked_values=sum(r['checked_values'] for r in rows),
            nan_values=sum(r['nan_values'] for r in rows),nan_payload_differences=payloads))
    return dict(passed=True,preparation=c.pin(c.BASE / 'prepared.json'),caller=variants,**observations)


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','complete']
    pre=preparation()
    if sys.argv[1]=='prepare':
        assert not (c.BASE / 'preparation-audit.json').exists()
        c.save(c.BASE / 'preparation-audit.json',pre);print(json.dumps({k:v for k,v in pre.items() if k not in ['identities','resources']}));return
    assert not (c.BASE / 'closed.json').exists()
    q=c.read(c.BASE / 'qualified.json');assert q['passed'];c.verify(q['files'])
    assert q['core']==c.pin(c.BASE / 'runtime/Lokad.Onnx.dll') and q['data']==c.pin(c.BASE / 'runtime/Lokad.Onnx.Data.dll')
    namespace=successor.configured()
    source=(c.TOOLS / 'qualify.py').read_text(encoding='utf8').replace('from prepare import *','')
    exec(compile(source,str(c.TOOLS / 'qualify.py'),'exec'),namespace)
    suite_results=[namespace['suite'](n,p,s) for n,_,_,p,s in namespace['SUITES']]
    assert suite_results==q['suites']==c.read(c.BASE / 'suites.json')
    jobs=[s[0] for s in namespace['SUITES']]+['package','consumer-restore','consumer-build','consumer']
    final=resources('qualification.json',jobs,[s[0] for s in namespace['SUITES']]+['consumer'])
    package=c.read(c.BASE / 'package.json');assert package['passed'] and package['package']==q['package']
    assert q['package']==c.pin(c.BASE / 'nuget/Lokad.Onnx.0.2.0.nupkg')
    consumer=c.read(c.BASE / 'consumer.json')
    assert consumer['passed'] and consumer['core']==q['core']['sha256'] and consumer['processor_count']==1
    assert consumer['packaged_narrow_values']==10240 and consumer['narrow_scratch_bytes']==327680
    assert consumer['packaged_tiled_convolution_values']==33216 and consumer['input_and_held_outputs_unchanged'] and consumer['model_imported']
    state=c.read(c.BASE / 'qualification.json');assert consumer['pid']==state['runs'][-1]['worker']['pid']
    identities=pre['identities']+final['identities'];records=pre['resources']+final['resources']
    analysis=dict(passed=True,core=q['core'],data=q['data'],package=q['package'],caller=pre['caller'],suites=suite_results,
        instruction_review=c.read(c.BASE / 'instruction-review.json'),identities=identities,resources=records,
        resource_samples=sum(r['samples'] for r in records),peak_rss=max(r['peak_rss'] for r in records),
        production_changed=False,models_qualified=False,performance_qualified=False)
    c.save(c.BASE / 'analysis.json',analysis)
    files=dict(c.read(c.BASE / 'prepared.json')['files']);files.update(q['files'])
    for folder in [c.BASE / 'logs',c.BASE / 'test-results']:
        files.update({c.rel(p):c.pin(p) for p in folder.rglob('*') if p.is_file()})
    for p in [*c.BASE.iterdir(),*c.TOOLS.iterdir()]:
        if p.is_file():files[c.rel(p)]=c.pin(p)
    c.save(c.BASE / 'closed.json',dict(passed=True,files=files,identities=identities,analysis=c.pin(c.BASE / 'analysis.json')))
    print(json.dumps(dict(closed=c.pin(c.BASE / 'closed.json'),resource_samples=analysis['resource_samples'])))


if __name__=='__main__':main()

"""Check every public call and resource sample, then summarize matched timings."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sys
from prepare import ROOT, BASE, TOOLS, INPUT, NATIVE, pin, save
from run import absent
sys.path.insert(0,str(ROOT/'tests/pyannote/natural-meetings'))
from audit import compare, inspect_public


def main():
    target=BASE/'closed.json';assert not target.exists()
    spec=json.loads((BASE/'prepared.json').read_text());controls=json.loads((BASE/'run-controls.json').read_text());limits=spec['limits']
    assert controls['prepared']==pin(BASE/'prepared.json') and controls['runner']==pin(TOOLS/'run.py') and controls['limits']==limits
    for name,wanted in spec['files'].items():assert pin(Path(name))==wanted,name
    state=json.loads((BASE/'processes.json').read_text());manifest=json.loads(INPUT.read_text())
    assert state['complete'] and state['code']==0 and absent(state['supervisor'])
    assert [r['role'] for r in state['runs']]==spec['jobs']
    identities=[state['supervisor']];samples=0;public=[];results=[]
    for index,run in enumerate(state['runs']):
        role=run['role'];assert run['index']==index and run['complete'] and run['code']==0 and absent(run['worker']);identities.append(run['worker'])
        folder=BASE/'process'/f'{index}-{role}';result=json.loads((folder/'output/result.json').read_text());results.append(result)
        assert result['schema']==1 and result['family']=='pyannote' and not result['conformance'] and result['affinity']==4 and result['held_outputs_unchanged']
        assert result['manifest_sha256']==pin(INPUT)['sha256']
        if role=='candidate':
            assert result['engine']=='managed' and result['runtime']=='.NET 10.0.12' and result['processor_count']==1 and not result['flags']
            for key,item in [('core_sha256','core'),('data_sha256','data'),('runner_sha256','consumer')]:assert result[key]==spec[item]['sha256']
            assert not (folder/'stderr.txt').read_text().strip()
        else:
            assert result['engine']=='ort' and result['onnxruntime']=='1.29.0' and result['numpy']=='2.2.4'
            assert result['native_binaries']==spec['native_binaries'] and result['native_settings']==spec['native_settings']
            assert result['runner_sha256']==pin(NATIVE)['sha256'] and result['adapter_sha256']==pin(NATIVE.with_name('native_adapters.py'))['sha256']
            assert result['python_binary_sha256']==pin(Path(sys.executable))['sha256']
            assert result['flags']=={k:'1' for k in ['MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}
        resource=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
        assert len(resource)==run['samples']>0 and max(s['rss'] for s in resource)==run['peak_rss'];samples+=len(resource)
        assert resource[-1]['seconds']<=run['seconds']<limits['seconds']
        for s in resource:
            assert s['rss']<limits['rss'] and s['available']>=limits['available'] and s['disk']>=limits['disk']
            assert s['affinity']==[2] and s['pid']==run['worker']['pid'] and s['birth']==run['worker']['birth']
        gaps=[resource[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(resource,resource[1:])]+[run['seconds']-resource[-1]['seconds']]
        assert all(0<=v<10 for v in gaps)
        preflight=[json.loads(s) for s in (folder/'preflight.jsonl').read_text().splitlines()]
        assert preflight[-1]==run['preflight'] and preflight[-1]['available']>=limits['preflight']
        assert all(s['seconds']<limits['preflight_wait_seconds'] and s['disk']>=limits['disk'] for s in preflight)
        assert len(result['records'])==16
        for number,row in enumerate(result['records']):
            case=manifest['cases'][number%4];iteration=number//4
            assert row==json.loads((folder/'output'/f'{number:03}.json').read_text())
            assert row['name']==case['name'] and row['pass']==iteration and row['phase']==('warmup' if iteration==0 else 'measured')
            assert row['ownership'] and row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']>0
            import numpy as np, hashlib
            pcm=np.load(ROOT/case['pcm']['path'],allow_pickle=False)
            assert row['input_sha256']==hashlib.sha256(pcm.tobytes()).hexdigest()
            inspect_public(row['result'],case['samples']);comparison=compare(row['result'],case['expected']);assert comparison['passed'],comparison
            public.append(dict(index=index,role=role,name=row['name'],iteration=iteration,**comparison))
    table=[];processes=[]
    with localcontext() as context:
        context.prec=50
        for case in manifest['cases']:
            row=dict(name=case['name'],audio_seconds=case['samples']/16000);means={}
            for role in ['candidate','ort']:
                values=[];visits=[]
                for index,(run,result) in enumerate(zip(state['runs'],results,strict=True)):
                    if run['role']!=role:continue
                    times=[Decimal(str(r['seconds'])) for r in result['records'] if r['name']==case['name'] and r['phase']=='measured']
                    assert len(times)==3;values+=times
                    visits.append(dict(index=index,mean=float(sum(times)/3),seconds=[float(v) for v in times]))
                assert len(values)==6;mean=sum(values)/6;means[role]=mean
                row[role]=dict(seconds=float(mean),rtf=float(mean/Decimal(str(row['audio_seconds']))),minimum=float(min(values)),maximum=float(max(values)),processes=visits)
            row['ratio']=float(means['candidate']/means['ort']);table.append(row)
        for index,(run,result) in enumerate(zip(state['runs'],results,strict=True)):
            processes.append(dict(index=index,role=run['role'],peak_rss=run['peak_rss'],means={c['name']:next(v['mean'] for v in c[run['role']]['processes'] if v['index']==index) for c in table}))
    analysis=dict(passed=True,calls=64,measured=48,warmup=16,table=table,processes=processes,public=public,
        samples=samples,identities=identities,maximum_centroid_error=max(c['maximum_centroid_error'] for c in public),
        limitation='Descriptive matched Windows comparison on active workstation; no calibrated confidence, AMD extrapolation or production promotion')
    save(BASE/'analysis.json',analysis)
    files=dict(spec['files'])
    for path in BASE.rglob('*'):
        if path.is_file():files[str(path.resolve())]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.resolve())]=pin(path)
    save(target,dict(passed=True,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps({k:v for k,v in analysis.items() if k!='public'}))
    print(json.dumps(dict(closure=pin(target),files=len(files))))


if __name__=='__main__':main()

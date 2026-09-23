"""Audit complete numerical census, cross-product identities, codegen and resources."""
import json
import re
from fixtures import verify_prefixes
from protocol import JOBS,LIMITS,pin,read,save,check_sample
from prepare import ROOT,BASE,previous_closed

def main():
    previous_closed();assert not (BASE/'closed.json').exists()
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(ROOT/name)==wanted,name
    folder=BASE/'collected';receipt=read(folder/'collection.json');state=read(folder/'identity.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert receipt['payload']==pin(BASE/'payload.json')
    payload=read(BASE/'payload.json')
    transfer=read(BASE/'collection-transfer.json');assert transfer['passed']
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(folder/'collection.json')
    assert prepared['archive']==pin(BASE/'payload.tar.gz') and prepared['stage']==pin(BASE/'bundle/stage.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json')
    assert state['ended']-state['started']<4*3600
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=0;peak=0;identities={(v['pid'],v['birth']) for v in receipt['identities']}
    assert (state['supervisor']['pid'],state['supervisor']['birth']) in identities
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        limit=LIMITS['build_preflight_available' if row['name'] in JOBS[:3] else 'preflight_available']
        assert row['preflight']['available']>=limit and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(line) for line in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and samples
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:
                assert (member['pid'],member['birth']) in identities
                assert row['members'][str(member['pid'])]==member['birth']
        assert max(s['rss'] for s in samples)==row['peak_rss']
        resources+=len(samples);peak=max(peak,row['peak_rss'])
    built=read(folder/'built.json');assert built['passed']
    assert (folder/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for name,wanted in built['files'].items():assert pin(folder/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items(): assert pin(folder/'runtimes'/role/name)==wanted
    results={};values={}
    shapes=[(51,1024,1024),(63,1024,1024),(64,1024,1024),(65,1024,1024),(66,1024,1024),
        (64,1023,1024),(64,1025,1024),(64,1024,1023),(64,1024,1025),(65,4096,1024),(66,1024,4096),(167,1024,1024),(225,1024,1024),(47,1024,1024),(48,1024,1024),(49,1024,1024),(50,1024,1024),(52,1024,1024),(61,1024,1024),(62,1024,1024),(48,1023,1024),(48,1025,1024),(48,1024,1023),(48,1024,1025),(49,1025,1025),(61,4096,1024),(61,1024,4096)]
    capture=read(BASE/'bundle/fixtures/result.json')
    assert verify_prefixes(read(BASE/'bundle/evidence/original-capture.json'),capture,BASE/'bundle/fixtures')
    census=[(f'fixture-{i}',r['m'],r['k'],r['n'],False,False) for i,r in enumerate(capture['entries'])]
    census += [(f'boundary-{i}',*shape,False,False) for i,shape in enumerate(shapes)]
    census += [(f'nonfinite-{m}',m,1024,1024,False,True) for m in [48,49,51,61,64,65,66]]
    census += [(f'scalar-{m}',m,1024,1024,True,False) for m in [48,49,51,61,63,64,65,66]]
    for run in state['runs'][3:]:
        name=run['name'];role,mode,width=name.split('-');value=read(folder/name/'result.json');values[name]=value
        assert value['completed'] and value['noPerformanceMeasurement'] and value['runtime']=='10.0.8'
        assert value['role']==role and value['mode']==mode and value['width']==int(width)
        assert value['assembly']==built['consumer']['sha256'] and value['pid']==run['child']['pid']
        assert value['core_sha256']==payload['products'][role]['Lokad.Onnx.dll']['sha256']
        flags={} if width=='512' else {'DOTNET_EnableAVX512':'0'}
        if mode=='codegen':flags['DOTNET_JitDisasm']='*RunFloatMatMulKernel* *RunGeneralFloatMatMulKernel* *RunShortWidePackedRows* *2x4packed_bump* *3x4packed*'
        assert value['flags']==flags
        rows=value['rows'];assert len(rows)==(66 if mode=='numerics' else 21)
        if mode=='numerics':
            for row,(label,m,n,k,scalar,nonfinite) in zip(rows,census):
                assert [row[f] for f in ['name','m','reduction','columns','scalar','nonfinite']]==[label,m,n,k,scalar,nonfinite]
                assert row['values']==m*k and row['independentCoordinates']==24
                assert all(row[f] is True for f in ['fullReference','inputs','guards','held'])
                assert row['scratch']==(2*n*k*4 if not scalar and (m>=64 or (role=='candidate' and m>=48 and n>=1024 and k>=1024)) else 0)
                assert len(row['mutatedOutputs'])==2
                for h in [row['output'],row['rawOutput'],*row['mutatedOutputs']]:assert re.fullmatch('[a-f0-9]{64}',h)
            for row,entry in zip(rows[:21],capture['entries']):assert row['output']==entry['y']['sha256']
            assert rows[63]==dict(name='alias-and-shape',refusals=4,unchanged=True)
            for row,label in zip(rows[64:],['broadcast-False','broadcast-True']):
                assert row['name']==label and row['reset']==row['failure']==3 and row['owned'] and row['inputs'] and len(row['outputs'])==3
        else:
            for row,entry in zip(rows,capture['entries']):
                assert [row[f] for f in ['name','node','m','reduction','columns']]==[entry[f] for f in ['name','node','m','k','n']]
                assert row['output']==entry['y']['sha256'] and row['calls']==80
            disasm=(folder/'logs'/(name+'.stdout')).read_text()
            assert 'Assembly listing for method Lokad.Onnx.Tensor' in disasm and 'RunFloatMatMulKernel' in disasm
            assert ' (Tier1)' in disasm
            assert all(method in disasm for method in ['3x4packed','2x4packed_bump'])
            assert 'ymm' in disasm and 'vfmadd' in disasm
        results[name]=dict(cases=len(rows),values=sum(r.get('values',0) for r in rows),full_equality=True)
    for mode,width in [('numerics',256),('numerics',512),('codegen',512)]:
        left=values[f'current-{mode}-{width}']['rows'];right=values[f'candidate-{mode}-{width}']['rows']
        # Scratch traffic is the intended difference; every semantic observation is exact.
        clean=lambda rows:[{k:v for k,v in row.items() if k!='scratch'} for row in rows]
        assert clean(left)==clean(right)
    for role in ['current','candidate']:
        assert values[f'{role}-numerics-256']['rows']==values[f'{role}-numerics-512']['rows']
    analysis=dict(passed=True,numerically_admitted=True,no_performance_measurement=True,root_product_changed=False,
        consumer=built['consumer'],products=payload['products'],resources=resources,peak_rss=peak,results=results,
        codegen_retained=True,codegen_review_required=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in folder.rglob('*') if p.is_file()}
    for name in ['analysis.json','prepared.json','staged.json','payload.json','deployment.json','collection-transfer.json','results.tar.gz','payload.tar.gz']:
        files[name]=pin(BASE/name)
    save(BASE/'closed.json',dict(passed=True,numerically_admitted=True,files=files,root_product_changed=False,no_performance_measurement=True))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),numerically_admitted=True,resources=resources,peak_rss=peak,results=results)))


if __name__=='__main__':main()

"""Recompute complete AMD references and reconcile all eight fresh replays."""
import copy
import json
from run import BASE,ROOT,prepared
from prepare import AMD,PLATFORM,adapted_source
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from checks import check_result,consumer_inventory


def references():
    import numpy as np
    fixtures=ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'
    old=read(fixtures/'output/result.json');expected=copy.deepcopy(old)
    native_source=read(PLATFORM/'collected/native-ort/result.json')
    reports=[];provenance=[];maximum=0.
    for ordinal,call in enumerate(expected['calls']):
        for slot,item in enumerate(call['outputs']):
            filename=str(ordinal).zfill(2)+'-'+str(slot)+'.f32';new_file='amd-'+filename
            selected=PLATFORM/'payload/retained-256'/filename;native=PLATFORM/'collected/native-ort'/filename
            row=native_source['reports'][ordinal*3+slot]
            assert row['pin']==pin(native) and row['shape']==item['shape']
            a=np.fromfile(selected,dtype='<f4').astype('float64');b=np.fromfile(native,dtype='<f4').astype('float64')
            assert a.shape==b.shape and a.size==item['values'] and np.isfinite(a).all() and np.isfinite(b).all()
            error=float((np.abs(a-b)/np.maximum(1.,np.abs(b))).max(initial=0));assert error<=1e-4
            call['outputs'][slot]=dict(item,file=new_file,**pin(selected))
            reports.append(dict(case=call['name'],index=call['index'],node=call['node'],slot=slot,
                reference=dict(file=new_file,shape=item['shape'],**pin(native)),
                comparison=dict(values=int(a.size),failed=0,maximum=error),exact_repeat=True,input_unchanged=True))
            prefix='/dev/shm/lokad-pyannote-lstm-platform-reference-v3-20260922/'
            provenance.append(dict(ordinal=ordinal,slot=slot,selected_source=prefix+'retained-256/'+filename,selected=pin(selected),native_source=prefix+'native-ort/'+filename,native=pin(native)))
            maximum=max(maximum,error)
        for item in call['inputs']:
            if item is not None:assert pin(fixtures/'output'/item['file'])=={k:item[k] for k in ['bytes','sha256']}
    expected['reference_provenance']=dict(platform='AMD EPYC 9V74, .NET10.0.8',original_capture=pin(fixtures/'output/result.json'),platform_closure=pin(PLATFORM/'collected/remote-closed.json'),operands_unchanged=True)
    expected_native=dict(passed=True,version='1.29.0',reports=reports,maximum=maximum,no_performance_measurement=True,source=pin(PLATFORM/'collected/native-ort/result.json'))
    assert read(BASE/'collected/references/original-capture.json')==old
    assert read(BASE/'collected/references/capture.json')==expected
    assert read(BASE/'collected/references/native.json')==expected_native
    assert read(BASE/'collected/references/provenance.json')==dict(passed=True,rows=provenance,original_payload=pin(AMD/'payload/payload.json'),platform_closure=pin(PLATFORM/'collected/remote-closed.json'),operands_unchanged=True)
    payload=read(BASE/'payload.json')
    for call in expected['calls']:
        for item in [*call['inputs'],*call['outputs']]:
            if item is not None:assert payload['files']['fixtures/output/'+item['file']]=={k:item[k] for k in ['bytes','sha256']}
    for row in reports:
        item=row['reference'];assert payload['files']['fixtures/native/'+item['file']]=={k:item[k] for k in ['bytes','sha256']}
    return expected,expected_native


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    collected=BASE/'collected';receipt=read(collected/'collection.json');transfer=read(BASE/'collection-transfer.json');payload=read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    assert (BASE/'bundle/source/consumer/ModelReplay.cs').read_text()==adapted_source()
    assert payload['files']['source/consumer/ModelReplay.cs']==pin(BASE/'bundle/source/consumer/ModelReplay.cs')
    capture,native=references();built=read(collected/'built.json');assert built['passed']
    for name,wanted in built['files'].items():assert pin(collected/name)==wanted,name
    for role,files in payload['products'].items():
        for name,wanted in files.items():assert pin(collected/'runtime'/role/name)==wanted,name
        assert pin(collected/'runtime'/role/'LstmModelReplay.dll')==built['consumer']
    inventory=consumer_inventory(read(collected/'inventory/instructions.json'),payload['previous_consumer'],built['consumer'])
    assert inventory==read(collected/'inventory/review.json')
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0 and state['boot_time']==1789634288.0
    assert state['supervisor']==read(BASE/'deployment.json') and [r['name'] for r in state['runs']]==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    results={};resources=[];actual=dict(payload,consumer=built['consumer'])
    for row in state['runs']:
        name=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']==row['preflight_observations'][-1] and row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (collected/'logs'/(name+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample);assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        if name.startswith(('selected-','candidate-')):
            role,width=name.split('-');result=read(collected/name/'result.json')
            assert result['pid']==row['child']['pid'] and result['runtime']=='10.0.8'
            results[name]=check_result(result,role,width,actual,BASE,capture,native)
            assert results[name]==read(collected/name/'review.json')
        resources.append(dict(name=name,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    assert (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    for width in ['256','512','scalar','simd']:assert results['selected-'+width]==results['candidate-'+width]
    analysis=dict(passed=True,products=payload['products'],consumer=built['consumer'],inventory=inventory,results=results,resources=resources,
        calls=192,outputs=576,values=29048832,inputs_unchanged=True,reference_provenance_verified=True,no_performance_measurement=True,codegen_pending=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()

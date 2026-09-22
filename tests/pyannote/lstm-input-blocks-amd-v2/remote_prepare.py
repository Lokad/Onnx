"""Derive AMD output references while preserving every captured input byte."""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
ORIGINAL=Path('/dev/shm/lokad-pyannote-lstm-input-blocks-20260922')
PLATFORM=Path('/dev/shm/lokad-pyannote-lstm-platform-reference-v3-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=12*1024**3 and psutil.disk_usage(BASE).free>=3*1024**3
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(ORIGINAL/'payload.json')['sha256']=='eab7aa40082b8662832791096b3e44bb71f396761294b2c775e106651922ae98'
    original=read(ORIGINAL/'payload.json')
    for name,wanted in original['files'].items():assert pin(ORIGINAL/name)==wanted,name
    assert pin(PLATFORM/'remote-closed.json')['sha256']=='50e487a962ed0916d15344bc6bfc21028c505fe68ecbcc4c5d94ca0ead860f46'
    closed=read(PLATFORM/'remote-closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(PLATFORM/name)==wanted,name
    for identity in closed['terminal_identities']:assert not live(identity)
    assert closed['terminal_identities'][0]==dict(pid=722132,birth=1790098268.9)
    parent=read(PLATFORM/'payload.json')
    for name,wanted in parent['external'].items():assert pin(name)==wanted,name
    shutil.copytree(ORIGINAL/'runtime',BASE/'runtime')
    shutil.copytree(ORIGINAL/'fixtures',BASE/'fixtures')
    (BASE/'references').mkdir()
    old=read(BASE/'fixtures/output/result.json');capture=copy.deepcopy(old)
    native=read(PLATFORM/'native-ort/result.json');reports=[];maximum=0.
    provenance=[]
    for ordinal,call in enumerate(capture['calls']):
        for slot,item in enumerate(call['outputs']):
            selected=PLATFORM/'retained-256'/(str(ordinal).zfill(2)+'-'+str(slot)+'.f32')
            row=native['reports'][ordinal*3+slot]
            assert (row['name'],row['index'],row['slot'])==(call['name'],call['index'],slot)
            reference=PLATFORM/'native-ort'/row['file'];assert pin(reference)==row['pin']
            assert row['shape']==item['shape'] and item['values']*4==pin(selected)['bytes']==pin(reference)['bytes']
            new_file='amd-'+str(ordinal).zfill(2)+'-'+str(slot)+'.f32'
            for source,folder in [(selected,BASE/'fixtures/output'),(reference,BASE/'fixtures/native')]:
                assert not (folder/new_file).exists();shutil.copy2(source,folder/new_file)
            a=np.fromfile(selected,dtype='<f4').astype('float64');b=np.fromfile(reference,dtype='<f4').astype('float64')
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            errors=np.abs(a-b)/np.maximum(1.,np.abs(b));error=float(errors.max(initial=0));assert error<=1e-4
            call['outputs'][slot]=dict(item,file=new_file,**pin(selected))
            reports.append(dict(case=call['name'],index=call['index'],node=call['node'],slot=slot,
                reference=dict(file=new_file,shape=item['shape'],**pin(reference)),
                comparison=dict(values=int(a.size),failed=0,maximum=error),exact_repeat=True,input_unchanged=True))
            provenance.append(dict(ordinal=ordinal,slot=slot,selected_source=str(selected),selected=pin(selected),native_source=str(reference),native=pin(reference)))
            maximum=max(maximum,error)
        a=copy.deepcopy(call);b=copy.deepcopy(old['calls'][ordinal]);a.pop('outputs');b.pop('outputs');assert a==b
    # All original files, including any shared input/output payloads, remain intact.
    for p in (ORIGINAL/'fixtures').rglob('*'):
        if p.is_file():assert pin(BASE/'fixtures'/p.relative_to(ORIGINAL/'fixtures'))==pin(p),p
    for call in capture['calls']:
        for item in call['inputs']:
            if item is not None:assert pin(BASE/'fixtures/output'/item['file'])=={k:item[k] for k in ['bytes','sha256']}
    capture['reference_provenance']=dict(platform='AMD EPYC 9V74, .NET10.0.8',original_capture=pin(ORIGINAL/'fixtures/output/result.json'),platform_closure=pin(PLATFORM/'remote-closed.json'),operands_unchanged=True)
    save(BASE/'fixtures/output/result.json',capture)
    new_native=dict(passed=True,version='1.29.0',reports=reports,maximum=maximum,no_performance_measurement=True,source=pin(PLATFORM/'native-ort/result.json'))
    save(BASE/'fixtures/native/result.json',new_native)
    for source,name in [(ORIGINAL/'fixtures/output/result.json','original-capture.json'),(BASE/'fixtures/output/result.json','capture.json'),(BASE/'fixtures/native/result.json','native.json')]:shutil.copy2(source,BASE/'references'/name)
    save(BASE/'references/provenance.json',dict(passed=True,rows=provenance,original_payload=pin(ORIGINAL/'payload.json'),platform_closure=pin(PLATFORM/'remote-closed.json'),operands_unchanged=True))
    assert len(capture['calls'])==12 and len(reports)==36 and maximum==1.591294694046004e-5
    assert original['cores']==stage['cores'] and original['consumer']==stage['consumer']
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=closed['terminal_identities'][0],boot_time=1789634288.0,
        cores=original['cores'],consumer=original['consumer'],external=parent['external'],interpreter=parent['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Exact AMD selected/candidate comparison with unchanged captured inputs, fixed native bound and private scratch; no timing claim.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),maximum=maximum)))


if __name__=='__main__':main()

"""Reuse pinned complete-model assets and unchanged measured products."""
import copy
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
PRODUCT=Path('/dev/shm/lokad-pyannote-lstm-input-product-v2-20260922')
APP=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(PRODUCT/'payload.json')['sha256']=='726202a539ab2919b36941f26a7eff8d5034cc84da41f9860a23dd9be12f7cee'
    product=read(PRODUCT/'payload.json');receipt=read(PRODUCT/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    assert product['measured']==stage['product']
    for name,wanted in product['files'].items():assert pin(PRODUCT/name)==wanted,name
    assert pin(APP/'payload.json')['sha256']=='229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'
    app=read(APP/'payload.json');previous=read(APP/'collection.json')
    assert previous['terminal'] and previous['code']==0 and previous['input_error'] is None
    for identity in previous['identities']:assert not live(identity)
    for name,wanted in app['files'].items():assert pin(APP/name)==wanted,name
    external=dict(product['external'])
    for name,wanted in app['external'].items():
        assert name not in external or external[name]==wanted
        external[name]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    for name in ['assets','graph-reference']:shutil.copytree(APP/name,BASE/name)
    assert pin(BASE/'graph-reference.json')==app['files']['graph-reference.json']
    (BASE/'manifests').mkdir();(BASE/'runtimes').mkdir();identities={};consumers={}
    for role in ['selected','candidate']:
        folder=BASE/'runtimes'/role;shutil.copytree(APP/'runtimes/portable',folder)
        if role=='candidate':
            for name,wanted in product['measured'].items():
                source=PRODUCT/'measured'/name;assert pin(source)==wanted;shutil.copy2(source,folder/name)
            for suffix in ['dll','deps.json','runtimeconfig.json']:(folder/('GraphQualification.'+suffix)).unlink()
        identities[role]={name:pin(folder/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        manifest=copy.deepcopy(read(BASE/'evidence/original-manifest.json'))
        assert manifest==read(APP/'manifests/portable-pyannote.json')
        manifest.update(core_sha256=identities[role]['Lokad.Onnx.dll']['sha256'],data_sha256=identities[role]['Lokad.Onnx.Data.dll']['sha256'])
        manifest['product_source']='selected M17 measured Core3c2f16b0/Data6318cf48' if role=='selected' else 'M22 measured Core208371f6/Datab9358370; four-row ordered LSTM input projection'
        save(BASE/'manifests'/(role+'-pyannote.json'),manifest)
        if role=='selected':
            assert pin(folder/'GraphQualification.dll')==stage['selected_consumer'];consumers[role]=stage['selected_consumer']
    assert (BASE/'consumer/Program.cs').read_bytes()==(APP/'graph-consumer/Program.cs').read_bytes().replace(stage['old_data'].encode(),stage['new_data'].encode())
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        identities=identities,consumers=consumers,old_data=stage['old_data'],new_data=stage['new_data'],
        external=external,interpreter=product['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Complete Pyannote graphs and sixteen public requests per measured role; diagnostic timings only.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()

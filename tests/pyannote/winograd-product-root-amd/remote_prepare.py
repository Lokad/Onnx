"""Use the closed qualified Winograd build and offline full-suite feed."""
import json,shutil,psutil
from pathlib import Path
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
BUILD=Path('/dev/shm/lokad-pyannote-winograd-product-build-v2-20260923')
APP=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(BUILD/'collection.json')==stage['build_collection']
    receipt=read(BUILD/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    for name,wanted in receipt['files'].items():assert pin(BUILD/name)==wanted,name
    assert pin(BUILD/'payload.json')==pin(BASE/'evidence/build-payload.json')
    previous=read(BUILD/'payload.json')
    for name,wanted in previous['files'].items():assert pin(BUILD/name)==wanted,name
    shutil.copytree(BUILD/'runtime',BASE/'measured')
    assert pin(BASE/'measured/Lokad.Onnx.dll')==stage['measured_core']
    assert pin(BASE/'measured/Lokad.Onnx.Data.dll')==stage['measured_data']
    assert pin(APP/'payload.json')['sha256']=='229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'
    app=read(APP/'payload.json');external=dict(previous['external'])
    for name,wanted in app['files'].items():
        if name.startswith('nuget-feed/'):
            source=APP/name;assert pin(source)==wanted;external[str(source)]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        measured={name:pin(BASE/'measured'/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},
        feed=str(APP/'nuget-feed'),external=external,interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Normal Winograd product, all original suites plus 17 added cases, both instruction widths and independent NuGet consumption; no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))

if __name__=='__main__':main()

"""Derive the measured candidate and offline feed from closed VM artifacts."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
SCREEN=Path('/dev/shm/lokad-pyannote-convolution-pointer-screen-20260923')
APP=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')
BUILD=Path('/dev/shm/lokad-pyannote-convolution-pointer-build-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(SCREEN/'payload.json')['sha256']=='ef37bd56afbbfb5c7f79526a62421848e122f5e8c21ba4f01b5a2d4abd030ac4'
    screen=read(SCREEN/'payload.json');receipt=read(SCREEN/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in screen['files'].items():assert pin(SCREEN/name)==wanted,name
    assert pin(SCREEN/'runtime/candidate/Lokad.Onnx.dll')==stage['measured_core']
    assert pin(BUILD/'collection.json')==stage['build_collection']
    build=read(BUILD/'collection.json')
    assert build['terminal'] and build['code']==0 and build['input_error'] is None
    assert all(not live(identity) for identity in build['identities'])
    assert pin(BUILD/'runtime/Lokad.Onnx.dll')==stage['measured_core']
    assert pin(BUILD/'runtime/Lokad.Onnx.Data.dll')==stage['measured_data']
    measured=BASE/'measured';measured.mkdir()
    for name,wanted in build['files'].items():
        if name.startswith('runtime/'):
            source=BUILD/name;assert pin(source)==wanted,name
            target=measured/Path(name).relative_to('runtime')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert pin(APP/'payload.json')['sha256']=='229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'
    app=read(APP/'payload.json');external=dict(screen['external'])
    for name,wanted in app['files'].items():
        if name.startswith('nuget-feed/'):
            source=APP/name;assert pin(source)==wanted;external[str(source)]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        measured={name:pin(measured/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},
        feed=str(APP/'nuget-feed'),external=external,interpreter=screen['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Normal Linux candidate build, exact compiled-method equivalence, full suites and actual NuGet consumption; no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()

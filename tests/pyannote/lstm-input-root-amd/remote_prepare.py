"""Derive the measured candidate and offline feed from closed VM artifacts."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
SCREEN=Path('/dev/shm/lokad-pyannote-lstm-input-screen-20260922')
APP=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(SCREEN/'payload.json')['sha256']=='3b05ac19410e48cd0b3baff83dfa5753a9123a162108d5846751b340ab0f452b'
    screen=read(SCREEN/'payload.json');receipt=read(SCREEN/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in screen['files'].items():assert pin(SCREEN/name)==wanted,name
    assert pin(SCREEN/'runtime/candidate/Lokad.Onnx.dll')==stage['measured_core']
    application=Path('/dev/shm/lokad-pyannote-lstm-input-app-20260922')
    assert pin(application/'payload.json')['sha256']=='1784c283a15fe441cc887f2b2fddc2232065252f9e7aeffc849cf10f37848591'
    application_receipt=read(application/'collection.json')
    assert application_receipt['terminal'] and application_receipt['code']==0 and application_receipt['input_error'] is None
    for identity in application_receipt['identities']:assert not live(identity)
    for name,wanted in application_receipt['files'].items():assert pin(application/name)==wanted,name
    assert pin(BASE/'evidence/application-closed.json')['sha256']=='73a4897a4db8e1bd729cb3c5486bcb11814d4ff669c9b9a472003572b08c64d0'
    assert pin(BASE/'evidence/application-analysis.json')==read(BASE/'evidence/application-closed.json')['analysis']
    assert read(BASE/'evidence/application-analysis.json')['performance']['admitted']
    measured=BASE/'measured';measured.mkdir()
    for name,wanted in screen['files'].items():
        if name.startswith('runtime/candidate/'):
            source=SCREEN/name;shutil.copy2(source,measured/source.name)
    assert pin(APP/'payload.json')['sha256']=='229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'
    app=read(APP/'payload.json');external=dict(screen['external'])
    for name,wanted in app['files'].items():
        if name.startswith('nuget-feed/'):
            source=APP/name;assert pin(source)==wanted;external[str(source)]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=application_receipt['identities'][0],boot_time=1789634288.0,
        measured={name:pin(measured/name) for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},
        feed=str(APP/'nuget-feed'),external=external,interpreter=screen['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Actual integrated root source on Linux, exact compiled-method equivalence, full suites and actual NuGet consumption; no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()

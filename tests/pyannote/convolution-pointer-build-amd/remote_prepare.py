"""Use complete selected dependencies and the already qualified AMD toolchain."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
CURRENT=Path('/dev/shm/lokad-parakeet-current-baseline-20260922')
PROFILE=Path('/dev/shm/lokad-pyannote-current-profile-v2-20260922')
ENVIRONMENT=Path('/dev/shm/lokad-pyannote-kernel-loop-build-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,evidence,key in [(CURRENT,'current','collected/collection.json'),
            (PROFILE,'profile','artifacts/pyannote-current-profile-amd-v2-20260922/collected/collection.json')]:
        assert pin(folder/'collection.json')==read(BASE/'evidence'/(evidence+'-closed.json'))['files'][key]
        receipt=read(folder/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        for identity in receipt['identities']:assert not live(identity)
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    proof=read(BASE/'evidence/profile-closed.json')
    assert pin(PROFILE/'exports/collection.json')==proof['files']['artifacts/pyannote-current-profile-amd-v2-20260922/exports/collection.json']
    receipt=read(PROFILE/'exports/collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in receipt['files'].items():assert pin(PROFILE/'exports'/name)==wanted,name
    shutil.copytree(CURRENT/'runtimes/current',BASE/'measured')
    for name,wanted in stage['measured'].items():assert pin(BASE/'measured'/name)==wanted
    assert (BASE/'measured/Google.Protobuf.dll').is_file()
    assert pin(ENVIRONMENT/'payload.json')['sha256']=='7bacf83b004b2c24d379b31b2a5923e14a504ef27b31a7f6004ad8a0c84a161c'
    previous=read(ENVIRONMENT/'payload.json')
    for name,wanted in previous['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        measured=stage['measured'],feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Isolated M28 CLI build; Kernel512 row-pointer and fixed-step traversal only; no timing or root integration.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()

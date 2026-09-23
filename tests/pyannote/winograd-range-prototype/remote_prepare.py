"""Pin existing fixtures/toolchain and require terminal preceding screen owners."""
from pathlib import Path
import json
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
SCREEN=Path('/dev/shm/lokad-pyannote-winograd-contiguous-screen-20260923')
ENVIRONMENT=Path('/dev/shm/lokad-pyannote-kernel-loop-build-20260922')
FIXTURES=Path('/dev/shm/lokad-pyannote-blocked-spatial-product-20260922/fixtures')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['build_preflight_available']
    assert psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(SCREEN/'collection.json')==pin(BASE/'evidence/screen-collection.json')
    receipt=read(SCREEN/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    assert pin(ENVIRONMENT/'payload.json')['sha256']=='7bacf83b004b2c24d379b31b2a5923e14a504ef27b31a7f6004ad8a0c84a161c'
    previous=read(ENVIRONMENT/'payload.json');external=dict(previous['external'])
    for name,wanted in stage['fixtures'].items():
        p=FIXTURES/name;assert pin(p)==wanted,name;external[str(p)]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        feed=previous['feed'],external=external,interpreter=previous['interpreter'],fixtures=str(FIXTURES),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Standalone M33 Winograd arithmetic numerics; selected direct source exact; no timing or root dispatch.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(external))))


if __name__=='__main__':main()

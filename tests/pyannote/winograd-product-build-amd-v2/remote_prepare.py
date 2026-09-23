"""Stage measured root runtime and verified dependencies from closed M33."""
import json,shutil,psutil
from pathlib import Path
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
CURRENT=Path('/dev/shm/lokad-parakeet-current-baseline-20260922')
SCREEN=Path('/dev/shm/lokad-pyannote-winograd-range-screen-20260923')

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(SCREEN/'collection.json')==pin(BASE/'evidence/screen-collection.json')
    receipt=read(SCREEN/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    for name,wanted in receipt['files'].items():assert pin(SCREEN/name)==wanted,name
    assert pin(SCREEN/'payload.json')==pin(BASE/'evidence/screen-payload.json')
    previous=read(SCREEN/'payload.json')
    for name,wanted in previous['external'].items():assert pin(name)==wanted,name
    measured=BASE/'measured';measured.mkdir()
    for name,wanted in stage['measured_files'].items():
        source=CURRENT/'runtimes/current'/name;assert pin(source)==wanted,name
        shutil.copy2(source,measured/name)
    for name,wanted in stage['measured'].items():assert pin(measured/name)==wanted
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        measured=stage['measured'],feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Isolated M34 product build and complete compiled-scope inventory; no timing or root integration.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))

if __name__=='__main__':main()

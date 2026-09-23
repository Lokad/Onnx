from pathlib import Path
import json,psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
CODE=Path('/dev/shm/lokad-pyannote-winograd-output-blocks-codegen-v2-20260923')

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['build_preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(CODE/'collection.json')==pin(BASE/'evidence/collection.json')
    receipt=read(CODE/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in receipt['files'].items():assert pin(CODE/name)==wanted,name
    assert pin(CODE/'payload.json')==pin(BASE/'evidence/payload.json')
    previous=read(CODE/'payload.json')
    for name,wanted in previous['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],previous_owner=receipt['identities'][0],boot_time=previous['boot_time'],
        feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],fixtures=previous['fixtures'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Fixed M36 whole-call screen; exact current and candidate product DLLs, identical Winograd boundary and all costs included.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))

if __name__=='__main__':main()

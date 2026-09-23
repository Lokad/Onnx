from pathlib import Path
import json,psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
PREVIOUS=Path('/dev/shm/lokad-release-graph-baseline-20260923')

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(PREVIOUS/'collection.json')==pin(BASE/'evidence/previous-collection.json')
    receipt=read(PREVIOUS/'collection.json');assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    for name,wanted in stage['external'].items():assert pin(name)==wanted,name
    previous=read(BASE/'evidence/previous-payload.json')
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=previous['boot_time'],
        feed=str(BASE/'empty-feed'),external=stage['external'],interpreter=stage['interpreter'],python_paths=stage['python_paths'],product=stage['product'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'})
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))

if __name__=='__main__':main()

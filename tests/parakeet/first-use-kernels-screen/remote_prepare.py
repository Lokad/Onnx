"""Check normal build closure and immutable dependencies before numerical workers."""
import json
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
BUILD=Path('/dev/shm/lokad-parakeet-first-use-kernels-build-20260923')
NUMERICS=Path('/dev/shm/lokad-parakeet-first-use-kernels-numerics-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['build_preflight_available']
    assert psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(BUILD/'collection.json')==pin(BASE/'evidence/build-collection.json')
    receipt=read(BUILD/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    assert pin(BUILD/'payload.json')==pin(BASE/'evidence/build-payload.json')
    previous=read(BUILD/'payload.json')
    for name,wanted in previous['files'].items():assert pin(BUILD/name)==wanted,name
    for name,wanted in previous['external'].items():assert pin(name)==wanted,name
    assert pin(NUMERICS/'collection.json')==pin(BASE/'evidence/numerics-collection.json')
    numeric_receipt=read(NUMERICS/'collection.json')
    assert numeric_receipt['terminal'] and numeric_receipt['code']==0 and numeric_receipt['input_error'] is None
    assert not any(live(i) for i in numeric_receipt['identities'])
    assert pin(NUMERICS/'payload.json')==pin(BASE/'evidence/numerics-payload.json')
    for name,wanted in read(NUMERICS/'payload.json')['files'].items():assert pin(NUMERICS/name)==wanted,name
    for name,wanted in numeric_receipt['files'].items():assert pin(NUMERICS/name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],previous_owner=numeric_receipt['identities'][0],boot_time=1789634288.0,
        feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Four fresh current/candidate/candidate/current processes; twenty-one fixtures, 60 warmups and 60 measurements per fixture; complete public destination call.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))


if __name__=='__main__':main()

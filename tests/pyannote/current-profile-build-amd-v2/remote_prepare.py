"""Reuse complete current and previous dependency sets; no product rebuild."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
CURRENT=Path('/dev/shm/lokad-parakeet-current-baseline-20260922')
PRIOR=Path('/dev/shm/lokad-pyannote-prepared-profile-v2-20260922')
SCREEN=Path('/dev/shm/lokad-pyannote-kernel-loop-screen-20260922')
FAILED=Path('/dev/shm/lokad-pyannote-current-profile-build-20260922')
BUILD=Path('/dev/shm/lokad-pyannote-kernel-loop-build-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,evidence,relative in [(CURRENT,'current',False),(PRIOR,'prior',True),(SCREEN,'screen',False)]:
        proof=read(BASE/'evidence'/(evidence+'-closed.json'))
        key=('artifacts/pyannote-prepared-profile-amd-v2-20260922/' if relative else '')+'collected/collection.json'
        assert pin(folder/'collection.json')==proof['files'][key]
        receipt=read(folder/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        for identity in receipt['identities']:assert not live(identity)
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    failure=read(BASE/'evidence/failure-closed.json')
    assert pin(FAILED/'collection.json')==failure['files']['collected/collection.json']
    receipt=read(FAILED/'collection.json');assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    for identity in receipt['identities']:assert not live(identity)
    for name,wanted in receipt['files'].items():assert pin(FAILED/name)==wanted,name
    assert pin(BUILD/'payload.json')['sha256']=='7bacf83b004b2c24d379b31b2a5923e14a504ef27b31a7f6004ad8a0c84a161c'
    environment=read(BUILD/'payload.json')
    shutil.copytree(CURRENT/'runtimes/current',BASE/'runtime')
    shutil.copytree(PRIOR/'runtime',BASE/'previous')
    assert not list((BASE/'runtime').glob('SampledAudio.*'))
    for name,wanted in stage['product'].items():assert pin(BASE/'runtime'/name)==wanted
    assert pin(BASE/'previous/SampledAudio.dll')==stage['previous_consumer']
    for name,wanted in environment['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,product=stage['product'],previous_consumer=stage['previous_consumer'],
        previous_owner=read(FAILED/'collection.json')['identities'][0],boot_time=1789634288.0,
        feed=environment['feed'],external=environment['external'],interpreter=environment['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Only current-product identity literals in SampledAudio; current product copied exactly, complete consumer instruction inventory required.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()

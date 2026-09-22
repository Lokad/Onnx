"""Stage unchanged diagnostic callers; ModelProbe.Main is never invoked."""
import json
from pathlib import Path
import shutil
import psutil
from protocol import JOBS,FILTER,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
NUM=Path('/dev/shm/lokad-pyannote-kernel-loop-numerics-v2-20260922')
CURRENT=Path('/dev/shm/lokad-parakeet-current-baseline-20260922')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for folder,digest in [(NUM,'1f8031a567f441db01302e997cd63235691454d82d5aecd94b40de7a500c5a40'),
                          (CURRENT,'8bdc588c30094ff248e393f38dbd2ab1dc2e1e06d811a415921783233c21b188')]:
        assert pin(folder/'payload.json')['sha256']==digest
        receipt=read(folder/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        for identity in receipt['identities']:assert not live(identity)
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    old=read(NUM/'payload.json');details={}
    for role in JOBS:
        target=BASE/'runtime'/role
        shutil.copytree(NUM/'consumers/layers/bin/Release/net10.0',target)
        assert pin(target/'Lokad.Onnx.dll')==stage['candidate_core'] and pin(target/'LayerGraphs.dll')==stage['probe']
        if role=='production':shutil.copy2(CURRENT/'runtimes/current/Lokad.Onnx.dll',target/'Lokad.Onnx.dll')
        wanted=stage['current_core'] if role=='production' else stage['candidate_core']
        assert pin(target/'Lokad.Onnx.dll')==wanted
        for p in (BASE/'driver').iterdir():shutil.copy2(p,target/p.name)
        details[role]=dict(core=wanted,probe=stage['probe'])
    for name,wanted in old['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,filter=FILTER,limits=LIMITS,previous_owner=read(NUM/'collection.json')['identities'][0],
        boot_time=1789634288.0,interpreter=old['interpreter'],external=old['external'],fixture_directory=old['fixture_directory'],
        driver=stage['driver'],job_details=details,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Unchanged diagnostic driver calls Fixtures/Candidate by reflection, never ModelProbe.Main. Both products run identical graph callers; no rebuild or timing score.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))


if __name__=='__main__':main()

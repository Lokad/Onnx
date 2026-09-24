"""Use actual integrated source, the flags-aware inventory and existing offline feed."""
import json,os,shutil,psutil
from pathlib import Path
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
BUILD=Path('/dev/shm/lokad-parakeet-validated-composition-models-20260924')
FEED=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922')
PRIOR=dict(build=BUILD,parakeet=Path('/dev/shm/lokad-parakeet-validated-composition-app-20260924'),
    pyannote=Path('/dev/shm/lokad-parakeet-validated-composition-pyannote-app-20260924'),
    graphs=Path('/dev/shm/lokad-parakeet-validated-composition-graphs-20260924'))

def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available'] and psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,folder in PRIOR.items():
        for name in ['payload.json','collection.json']:assert pin(folder/name)==pin(BASE/'evidence'/label/name)
        receipt=read(folder/'collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for name,wanted in read(folder/'payload.json')['files'].items():assert pin(folder/name)==wanted,name
    previous=read(BUILD/'payload.json');shutil.copytree(BUILD/'runtimes/candidate',BASE/'measured',copy_function=os.link)
    for name,wanted in stage['measured'].items():assert pin(BASE/'measured'/name)==wanted
    assert pin(FEED/'payload.json')['sha256']=='229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'
    external=dict(previous['external'])
    for name,wanted in read(FEED/'payload.json')['files'].items():
        if name.startswith('nuget-feed/'):
            source=FEED/name;assert pin(source)==wanted;external[str(source)]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        measured=stage['measured'],feed=str(FEED/'nuget-feed'),external=external,interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='425actual rootfiles; all methodbodies/implementationflags/API equal to measured M66; fullbackend/tensor suites at bothwidths, actualNuGetPackageReference.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']))))

if __name__=='__main__':main()

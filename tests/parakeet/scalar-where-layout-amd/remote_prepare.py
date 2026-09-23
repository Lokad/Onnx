"""Bind terminal owners, selected DLLs, offline feed and existing model assets."""
import json
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
PARENTS=dict(build='/dev/shm/lokad-parakeet-wide-entry-first-use-build-20260923',
    models='/dev/shm/lokad-parakeet-wide-entry-first-use-models-20260923',
    release='/dev/shm/lokad-parakeet-wide-entry-first-use-root-v2-20260923',
    prior='/dev/shm/lokad-parakeet-ordered-wide-blocks-screen-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    for label,name in PARENTS.items():
        parent=Path(name)
        for file in ['collection.json','payload.json']:
            assert pin(parent/file)==pin(BASE/'evidence'/(label+'-'+file))
        receipt=read(parent/'collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        assert not any(live(i) for i in receipt['identities'])
        for file,wanted in read(parent/'payload.json')['files'].items():assert pin(parent/file)==wanted,file
    previous=read(Path(PARENTS['build'])/'payload.json')
    external=dict(previous['external'])
    for name,wanted in stage['models'].items():
        assert name not in external or external[name]==wanted
        external[name]=wanted
    for name,wanted in external.items():assert pin(name)==wanted,name
    for name,wanted in stage['product'].items():assert pin(BASE/'runtimes/current'/name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,product=stage['product'],
        previous_owner=receipt['identities'][0],boot_time=1789634288.0,feed=previous['feed'],
        external=external,interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        scope='Unchanged selected encoder with logger callback; two complete outputs,146Where observations,eight fixed fixtures; no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'))))


if __name__=='__main__':main()

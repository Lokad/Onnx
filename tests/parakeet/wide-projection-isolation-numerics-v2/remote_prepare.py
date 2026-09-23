"""Check normal build closure and immutable dependencies before numerical workers."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
BUILD=Path('/dev/shm/lokad-parakeet-wide-projection-isolation-build-20260923')


def main():
    psutil.Process().cpu_affinity([0]);idle();assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['build_preflight_available']
    assert psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json')
    fixture_receipt=Path('/dev/shm/lokad-parakeet-isolated-short-kernels-screen-20260923/collection.json')
    assert pin(fixture_receipt)==pin(BASE/'evidence/fixture-collection.json')
    retained=read(fixture_receipt)
    assert retained['terminal'] and retained['code']==0 and retained['input_error'] is None
    assert all(not live(identity) for identity in retained['identities'])
    for name,link in stage['fixture_links'].items():
        target=(BASE/name).resolve();assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source=Path(link['source']);assert pin(source)==link['identity']==stage['files'][name]
        target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    assert pin(BUILD/'collection.json')==pin(BASE/'evidence/build-collection.json')
    receipt=read(BUILD/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    assert pin(BUILD/'payload.json')==pin(BASE/'evidence/build-payload.json')
    previous=read(BUILD/'payload.json')
    for name,wanted in previous['files'].items():assert pin(BUILD/name)==wanted,name
    for name,wanted in previous['external'].items():assert pin(name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Actual current/candidate DLL numerical qualification and complete generated code; no timing.')
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))


if __name__=='__main__':main()

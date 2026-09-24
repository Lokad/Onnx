"""Prove the terminal control and its immutable inputs before diagnostic execution."""
import json
import os
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,DIAGNOSTIC_FLAGS,pin,read,save,verify
from remote import idle,live

BASE=Path(__file__).resolve().parents[1]
CONTROL=Path('/dev/shm/lokad-parakeet-dense-scalar-where-stability-20260924')


def main():
    psutil.Process().cpu_affinity([0]); idle(); assert not (BASE/'payload.json').exists()
    assert psutil.virtual_memory().available>=LIMITS['preflight_available']
    assert psutil.disk_usage(BASE).free>=LIMITS['preflight_tmpfs']
    stage=read(BASE/'stage.json'); assert stage['products']['current']==stage['products']['candidate']
    assert stage['diagnostic_flags']==DIAGNOSTIC_FLAGS
    fixture_receipt=Path('/dev/shm/lokad-parakeet-scalar-where-layout-20260923/collection.json')
    assert pin(fixture_receipt)==pin(BASE/'evidence/fixture-collection.json')
    retained=read(fixture_receipt)
    assert retained['terminal'] and retained['code']==0 and retained['input_error'] is None
    assert all(not live(identity) for identity in retained['identities'])
    for name,link in stage['fixture_links'].items():
        target=(BASE/name).resolve(); assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source=Path(link['source']); assert pin(source)==link['identity']==stage['files'][name]
        target.parent.mkdir(parents=True,exist_ok=True); os.link(source,target)
    for name,wanted in stage['files'].items(): assert pin(BASE/name)==wanted,name
    assert pin(CONTROL/'collection.json')==pin(BASE/'evidence/control-collection.json')
    receipt=read(CONTROL/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert all(not live(identity) for identity in receipt['identities'])
    assert pin(CONTROL/'payload.json')==pin(BASE/'evidence/control-payload.json')
    previous=read(CONTROL/'payload.json')
    for name,wanted in previous['files'].items(): assert pin(CONTROL/name)==wanted,name
    for name,wanted in previous['external'].items(): assert pin(name)==wanted,name
    built=read(BASE/'built.json'); assert built['passed'] and built['consumer']==stage['consumer']
    assert pin(BASE/'built.json')==pin(CONTROL/'built.json')
    for name,wanted in built['files'].items(): assert pin(BASE/name)==pin(CONTROL/name)==wanted,name
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],consumer=stage['consumer'],
        diagnostic_flags=DIAGNOSTIC_FLAGS,previous_owner=receipt['identities'][0],boot_time=1789634288.0,
        feed=previous['feed'],external=previous['external'],interpreter=previous['interpreter'],
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'},
        scope='Exact frozen consumer and identical selected products; all 220 cases, 600 warmup/180 measured intervals; only JitDisasm added. Diagnostic only, no score.')
    save(BASE/'payload.json',payload); verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'),files=len(payload['files']),external=len(payload['external']))))


if __name__=='__main__': main()

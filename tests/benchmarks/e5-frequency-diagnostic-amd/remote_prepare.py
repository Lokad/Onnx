"""Link existing exact products; pin the installed external counter binaries."""
import json
import os
import sys
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]


def main():
    idle();assert psutil.boot_time()==1789634288.0 and not (BASE/'payload.json').exists()
    stage=read(BASE/'stage.json')
    assert psutil.virtual_memory().available>=LIMITS['preflight_available']+stage['storage_estimate']['initial_reserve']
    for name,wanted in stage['files'].items():assert pin(BASE/name)==wanted,name
    receipt=read(BASE/'evidence/original-collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert not any(live(i) for i in receipt['identities'])
    assert pin(stage['receipts']['original'])==pin(BASE/'evidence/original-collection.json')
    for name,link in stage['links'].items():
        target=(BASE/name).resolve();assert target.is_relative_to(BASE.resolve()) and not target.exists()
        source=Path(link['source']);assert pin(source)==link['identity'],str(source)
        target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
    external=dict(stage['external'])
    for binary in ['/usr/bin/perf','/usr/bin/sudo',sys.executable]:external[binary]=pin(binary)
    payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],
        previous_owner=receipt['identities'][0],previous_consumer=stage['previous_consumer'],
        boot_time=psutil.boot_time(),external=external,interpreter=stage['interpreter'],
        storage_estimate=stage['storage_estimate'],clock_proof=stage['clock_proof'],original_closure=stage['original_closure'],
        failed_release_controls=stage['failed_release_controls'],diagnostic_only=True,release_admitted=False,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()})
    save(BASE/'payload.json',payload);verify(BASE)
    print(json.dumps(dict(passed=True,payload=pin(BASE/'payload.json'))))


if __name__=='__main__':main()

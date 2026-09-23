"""Link only immutable, identity-checked inputs from terminal prior work."""
import os,sys
from pathlib import Path
import psutil
from protocol import JOBS,LIMITS,pin,read,save,verify
from remote import idle,live
BASE=Path(__file__).resolve().parents[1]
idle();assert psutil.boot_time()==1789634288.0 and not (BASE/'payload.json').exists()
stage=read(BASE/'stage.json');receipt=read(BASE/'evidence/graph-collection.json')
assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt['identities'])
assert pin(Path('/dev/shm/lokad-parakeet-first-use-kernels-graphs-20260923/collection.json'))==pin(BASE/'evidence/graph-collection.json')
for n,v in stage['files'].items():assert pin(BASE/n)==v,n
for n,link in stage['links'].items():
    target=(BASE/n).resolve();assert target.is_relative_to(BASE.resolve()) and not target.exists()
    source=Path(link['source']);assert pin(source)==link['identity'],str(source)
    target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
payload=dict(passed=True,jobs=JOBS,limits=LIMITS,products=stage['products'],previous_owner=receipt['identities'][0],
    boot_time=psutil.boot_time(),feed=stage['feed'],external=stage['external'],interpreter=stage['interpreter'],
    diagnostic_only=True,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()})
save(BASE/'payload.json',payload);verify(BASE)
print('Baseline-only diagnostic prepared')

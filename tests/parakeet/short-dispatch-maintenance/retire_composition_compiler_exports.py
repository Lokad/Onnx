"""Retire seven terminal VM compiler exports with complete verified local copies."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-runtime-diagnostic-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
OUT=ROOT/'artifacts/e5-diagnostic-compiler-retention-20260924'
SUPERVISOR=dict(pid=954546,birth=1790264329.59)
TARGETS=[
    ('parakeet-prepared-recurrence-build-amd-20260924','lokad-parakeet-prepared-recurrence-build-20260924','collected','collection.json'),
    ('parakeet-short-dispatch-build-amd-20260923','lokad-parakeet-short-dispatch-build-20260923','collected','collection.json'),
    ('parakeet-slice-layout-amd-20260924','lokad-parakeet-slice-layout-20260924','build-collected','build-collection.json'),
    ('parakeet-slice-materialization-build-amd-20260924','lokad-parakeet-slice-materialization-build-20260924','build-collected','build-collection.json'),
    ('parakeet-slice-materialization-build-amd-v2-20260924','lokad-parakeet-slice-materialization-build-v2-20260924','build-collected','build-collection.json'),
    ('parakeet-validated-composition-build-amd-20260924','lokad-parakeet-validated-composition-build-20260924','build-collected','build-collection.json'),
    ('parakeet-validated-composition-build-amd-v2-20260924','lokad-parakeet-validated-composition-build-v2-20260924','build-collected','build-collection.json'),
]


def main():
    assert not OUT.exists();targets=[]
    for local,remote,collection,receipt_name in TARGETS:
        root=ROOT/'artifacts'/local;folder=root/collection;receipt=read(folder/receipt_name)
        assert receipt['terminal']
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
        wanted=receipt['files']['inventory/instructions.json']
        if 'identities' in receipt:identities=receipt['identities']
        else:
            state=read(folder/'build-state.json');assert state['complete']
            identities=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
        targets.append(dict(root='/dev/shm/'+remote,name='inventory/instructions.json',identity=wanted,
            collection=receipt_name,collection_identity=pin(folder/receipt_name),identities=identities,
            local=(folder/'inventory/instructions.json').relative_to(ROOT).as_posix()))
    result=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read
from remote import live
assert psutil.boot_time()==1789634288.0
def between_workers():
 state=read(base/'identity.json');assert state['supervisor']=={SUPERVISOR!r}
 assert all(r['complete'] and r['code']==0 for r in state['runs'])
 assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
 own=psutil.Process();allowed={{own.pid,*[p.pid for p in own.parents()],{SUPERVISOR['pid']}}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in allowed:continue
  command=' '.join(p.info['cmdline'] or [])
  assert p.info['name'] not in ['dotnet','perf'],p.info
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command),p.info
 return dict(supervisor_live=live(state['supervisor']),completed=len(state['runs']),complete=state['complete'])
first=between_workers();targets={targets!r};protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  if not (folder/name).exists():continue
  value=read(folder/name)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{{}}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{{}}))
for row in targets:
 root=Path(row['root']);target=root/row['name']
 assert root.resolve().parent==Path('/dev/shm') and target.resolve().is_relative_to(root.resolve()) and target.resolve()==target
 assert not target.is_symlink() and target.stat().st_nlink==1
 assert not any(live(i) for i in row['identities'])
 assert pin(root/row['collection'])==row['collection_identity']
 assert str(target) not in protected and pin(target)==row['identity']
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
last=between_workers();assert last==first
for row in targets:(Path(row['root'])/row['name']).unlink()
assert all(not (Path(row['root'])/row['name']).exists() for row in targets)
print(json.dumps(dict(passed=True,retired=targets,bytes_retired=sum(r['identity']['bytes'] for r in targets),
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 protected_paths=len(protected),between_workers=last)))
'''))
    for row in targets:assert pin(ROOT/row['local'])==row['identity']
    OUT.mkdir();save(OUT/'closed.json',dict(**result,generator=pin(Path(__file__))))
    print(json.dumps({k:v for k,v in result.items() if k!='retired'}))


if __name__=='__main__':main()

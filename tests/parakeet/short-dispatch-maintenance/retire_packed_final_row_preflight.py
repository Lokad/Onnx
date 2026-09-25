"""Retire a completed restore cache and exact VM copies of closed raw probes."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/packed-final-row-build'))
from run import ssh,PRELUDE,pin,read,write

BASE=ROOT/'artifacts/parakeet-packed-final-row-preflight-retention-20260925'
TARGETS=[
    ('parakeet-packed-final-row-build-amd-20260925','lokad-parakeet-packed-final-row-build-20260925',
     '6295ad30835b7a2a1694b580a8e1447a6a0cdb28828a5960fa6e0e7c3628576f','packages'),
    ('parakeet-owned-packed-weight-reconstruction-cost-amd-20260925','lokad-parakeet-owned-packed-weight-reconstruction-cost-20260925',
     '1750ff9e337a06840f1001b335ac211909bc721c09abc33f8aabc389e46c8f38','probe')]


def main():
    assert not BASE.exists();roots={}
    for local,remote,digest,kind in TARGETS:
        folder=ROOT/'artifacts'/local
        assert pin(folder/'closed.json')['sha256']==digest
        closure=read(folder/'closed.json')
        for name,wanted in closure['files'].items():assert pin(folder/name)==wanted,name
        collected=folder/'capture-collected';receipt=read(collected/'capture-collection.json')
        assert receipt['terminal'] and receipt['code']==0
        copies={n:w for n,w in receipt['files'].items() if n.startswith('probe/')} if kind=='probe' else {}
        assert kind!='probe' or len(copies)==324
        roots['/dev/shm/'+remote]=dict(kind=kind,local=local,collection=pin(collected/'capture-collection.json'),
            state=pin(collected/'capture-state.json'),copies=copies)
    common=PRELUDE+'''
from remote import idle,live,read,pin
idle();assert psutil.boot_time()==1789634288.0
protected=set();manifests={}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  p=folder/name
  if not p.is_file():continue
  value=read(p);manifests[str(p)]=pin(p)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{}))
'''
    snapshot=ssh(common+f'''
roots={roots!r};files={{}};retained={{}};owners=[];archives=0
for name,wanted in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'capture-collection.json')==wanted['collection'] and pin(root/'capture-state.json')==wanted['state']
 receipt=read(root/'capture-collection.json');state=read(root/'capture-state.json')
 assert receipt['terminal'] and state['complete'] and receipt['code']==state['code']==0
 assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
 scope=root/wanted['kind'];assert scope.is_dir() and not scope.is_symlink()
 paths=[]
 for p in scope.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():paths.append(p)
 assert paths
 if wanted['kind']=='probe':assert {{p.relative_to(root).as_posix() for p in paths}}==set(wanted['copies'])
 for p in paths:
  assert p.resolve()==p and p.is_relative_to(scope) and str(p) not in protected and p.stat().st_nlink==1
  identity=pin(p);files[str(p)]=identity
  if wanted['kind']=='probe':assert identity==wanted['copies'][p.relative_to(root).as_posix()]
  elif p.suffix=='.nupkg':
   original=Path(read(root/'spec.json')['feed'])/p.name
   assert original.resolve()==original and original.is_file() and pin(original)==identity
   retained[str(original)]=identity;archives+=1
assert archives==23
print(json.dumps(dict(passed=True,roots=roots,owners=owners,files=files,retained=retained,archives=archives,manifests=manifests,
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''')
    BASE.mkdir();write(BASE/'prospective.json',snapshot)
    result=ssh(common+f'''
value={snapshot!r}
assert manifests==value['manifests'] and not any(live(i) for i in value['owners'])
for name,wanted in value['roots'].items():
 root=Path(name);assert pin(root/'capture-collection.json')==wanted['collection'] and pin(root/'capture-state.json')==wanted['state']
 scope=root/wanted['kind']
 assert {{str(p) for p in scope.rglob('*') if p.is_file()}}=={{n for n in value['files'] if Path(n).is_relative_to(scope)}}
for name,wanted in value['files'].items():
 p=Path(name);assert p.resolve()==p and not p.is_symlink() and p.stat().st_nlink==1 and name not in protected
 assert any(p.is_relative_to(Path(r)/v['kind']) for r,v in value['roots'].items()) and pin(p)==wanted
for name,wanted in value['retained'].items():assert name not in value['files'] and pin(name)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
physical=sum(Path(n).stat().st_blocks*512 for n in value['files'])
for name in value['files']:Path(name).unlink()
assert all(not Path(n).exists() for n in value['files'])
for name,wanted in value['retained'].items():assert pin(name)==wanted
idle();assert not any(live(i) for i in value['owners'])
print(json.dumps(dict(passed=True,files=len(value['files']),logical_bytes=sum(v['bytes'] for v in value['files'].values()),
 physical_bytes=physical,archives=value['archives'],before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''')
    for wanted in roots.values():
        for name,identity in wanted['copies'].items():assert pin(ROOT/'artifacts'/wanted['local']/'capture-collected'/name)==identity
    write(BASE/'closed.json',dict(**result,prospective=pin(BASE/'prospective.json'),script=pin(Path(__file__)),
        scope='One terminal generated restore cache and 324 exact VM probe duplicates; offline packages and complete local raw results retained.'))
    print(json.dumps(result))


if __name__=='__main__':main()

"""Retire only the closed M63 product and focused-test package caches."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-root-amd-v2'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-inclusive-packing-package-retention-20260924'
TARGETS=[
    ('parakeet-inclusive-packing-build-amd-20260924','lokad-parakeet-inclusive-packing-build-20260924',
     '50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078',0),
    ('parakeet-inclusive-packing-contracts-amd-v2-20260924','lokad-parakeet-inclusive-packing-contracts-v2-20260924',
     'e85e5f5f7d435141b41b19957cc9bc119850c0593a120956002c798e66d01b76',0)]



def main():
    assert not BASE.exists();roots={}
    for local,remote,digest,code in TARGETS:
        folder=ROOT/'artifacts'/local;assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']==(code==0)
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        package=folder/'collected/nuget/Lokad.Onnx.0.2.0.nupkg'
        roots['/dev/shm/'+remote]=dict(code=code,collection=pin(folder/'collected/collection.json'),
            package=pin(package) if package.is_file() else None)
    BASE.mkdir()
    common=PRELUDE+'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 if not (folder/'payload.json').is_file():continue
 spec=read(folder/'payload.json')
 protected.update(str((folder/n).resolve()) for n in spec.get('files',{}))
 protected.update(str(Path(n).resolve()) for n in spec.get('external',{}))
'''
    snapshot=json.loads(ssh(common+f'''
roots={roots!r};files={{}};retained={{}};owners=[];archives=0
for name,wanted in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'collection.json')==wanted['collection']
 receipt=read(root/'collection.json')
 assert receipt['terminal'] and receipt['code']==wanted['code'] and receipt['input_error'] is None
 assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
 spec=read(root/'payload.json')
 for n,w in spec['files'].items():assert pin(root/n)==w,n
 for n,w in spec['external'].items():assert pin(Path(n))==w,n
 cache=root/'packages';assert cache.is_dir() and not cache.is_symlink()
 paths=[]
 for p in cache.rglob('*'):
  assert not p.is_symlink(),str(p)
  if p.is_file():paths.append(p)
 assert paths
 for p in paths:
  assert p.resolve().is_relative_to(cache) and str(p.resolve()) not in protected
  files[str(p)]=pin(p)
  if p.suffix=='.nupkg':
   original=Path(spec['feed'])/p.name
   if p.name.lower()=='lokad.onnx.0.2.0.nupkg':
    original=root/'nuget/Lokad.Onnx.0.2.0.nupkg'
    assert pin(original)==wanted['package']==receipt['files']['nuget/Lokad.Onnx.0.2.0.nupkg']
   else:assert str(original.resolve()) in protected
   assert original.is_file() and pin(original)==files[str(p)]
   retained[str(original)]=pin(original);archives+=1

assert archives>0
print(json.dumps(dict(passed=True,roots=roots,owners=owners,files=files,retained=retained,archives=archives,
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    save(BASE/'prospective.json',snapshot)
    result=json.loads(ssh(common+f'''
value={snapshot!r}
assert not any(live(i) for i in value['owners'])
for name,wanted in value['roots'].items():
 root=Path(name);assert pin(root/'collection.json')==wanted['collection']
 paths={{str(p) for p in (root/'packages').rglob('*') if p.is_file()}}
 expected={{n for n in value['files'] if Path(n).is_relative_to(root/'packages')}}
 assert paths==expected
for name,wanted in value['files'].items():
 p=Path(name);assert not p.is_symlink() and str(p.resolve())==name and name not in protected
 assert any(p.is_relative_to(Path(r)/'packages') for r in value['roots'])
 assert pin(p)==wanted,name
for name,wanted in value['retained'].items():assert name not in value['files'] and pin(name)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in value['files']:Path(name).unlink()
assert all(not Path(name).exists() for name in value['files'])
for name,wanted in value['retained'].items():assert pin(name)==wanted
for name in value['roots']:
 root=Path(name);spec=read(root/'payload.json')
 for n,w in spec['files'].items():assert pin(root/n)==w,n
 for n,w in spec['external'].items():assert pin(Path(n))==w,n
idle();assert not any(live(i) for i in value['owners'])
print(json.dumps(dict(passed=True,files=len(value['files']),logical_bytes=sum(v['bytes'] for v in value['files'].values()),
 archives=value['archives'],before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    save(BASE/'closed.json',dict(**result,prospective=pin(BASE/'prospective.json'),script=pin(__file__),
        scope='Two terminal generated package caches only; every package archive matches its retained original. All immutable inputs, products and complete local proofs retained.'))
    print(json.dumps(result))


if __name__=='__main__':main()

"""Retire a closed generated cache and the fully retained inspection export."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/observed-dense-where-inventory-amd'))
from run import ssh, PRELUDE
from prepare import previous_closed
from protocol import pin, read, save
BUILD=ROOT/'artifacts/parakeet-observed-dense-where-build-amd-20260924'
REVIEW=ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
OUT=ROOT/'artifacts/parakeet-observed-where-build-retention-20260924'
REMOTE='/dev/shm/lokad-parakeet-observed-dense-where-build-20260924'
REVIEW_REMOTE='/dev/shm/lokad-parakeet-observed-dense-where-inventory-20260924'
FEED='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed'


def main():
    assert not OUT.exists(); previous_closed()
    assert pin(REVIEW/'closed.json')['sha256']=='7d5296b266feeacbf9c52bf57a689de2cdf0a839045130f15912912f5da1b7bd'
    proof=read(REVIEW/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(REVIEW/name)==wanted,name
    roots={}
    for folder,remote in [(BUILD,REMOTE),(REVIEW,REVIEW_REMOTE)]:
        receipt=read(folder/'collected/collection.json')
        roots[remote]=dict(collection=pin(folder/'collected/collection.json'),owners=receipt['identities'],
                          payload=pin(folder/'payload.json'),code=receipt['code'])
    export=pin(REVIEW/'collected/inventory/instructions.json')
    common=PRELUDE+f'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
roots={roots!r}
for name,value in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'collection.json')==value['collection'] and pin(root/'payload.json')==value['payload']
 receipt=read(root/'collection.json');assert receipt['terminal'] and receipt['code']==value['code']
 assert receipt['input_error'] is None and not any(live(i) for i in value['owners'])
 spec=read(root/'payload.json')
 for n,w in spec['files'].items():assert pin(root/n)==w,n
 for n,w in spec['external'].items():assert pin(n)==w,n
cache=Path({REMOTE!r})/'packages';export=Path({REVIEW_REMOTE!r})/'inventory/instructions.json'
feed=Path({FEED!r})
assert cache.is_dir() and not cache.is_symlink() and cache.resolve()==cache
assert export.resolve()==export and not export.is_symlink() and pin(export)=={export!r}
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  if not (folder/name).is_file():continue
  value=read(folder/name)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{{}}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{{}}))
'''
    snapshot=json.loads(ssh(common+'''
files={};retained={}
for p in [*(p for p in cache.rglob('*') if p.is_file()),export]:
 assert not p.is_symlink() and p.resolve()==p and (p.is_relative_to(cache) or p==export)
 assert str(p) not in protected
 files[str(p)]=pin(p)
 if p.suffix=='.nupkg':
  original=feed/p.name;assert str(original.resolve()) in protected and pin(original)==files[str(p)]
  retained[str(original)]=pin(original)
assert retained
print(json.dumps(dict(files=files,retained=retained,protected_paths=len(protected))))
''',300))
    OUT.mkdir();save(OUT/'prospective.json',snapshot)
    result=json.loads(ssh(common+f'''
value={snapshot!r}
assert {{str(p) for p in cache.rglob('*') if p.is_file()}}|{{str(export)}}==set(value['files'])
for name,wanted in value['files'].items():
 p=Path(name);assert not p.is_symlink() and p.resolve()==p and (p.is_relative_to(cache) or p==export)
 assert name not in protected and pin(p)==wanted,name
for name,wanted in value['retained'].items():assert name not in value['files'] and pin(name)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in value['files']:Path(name).unlink()
assert all(not Path(name).exists() for name in value['files'])
for name,wanted in value['retained'].items():assert pin(name)==wanted
for name,value in roots.items():
 root=Path(name);spec=read(root/'payload.json')
 for n,w in spec['files'].items():assert pin(root/n)==w,n
 for n,w in spec['external'].items():assert pin(n)==w,n
 assert not any(live(i) for i in value['owners'])
idle()
print(json.dumps(dict(passed=True,files=len({snapshot['files']!r}),archives=len({snapshot['retained']!r}),
 logical_bytes={sum(v['bytes'] for v in snapshot['files'].values())},before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    save(OUT/'closed.json',dict(**result,source_closure=pin(REVIEW/'closed.json'),retained_export=export,
         prospective=pin(OUT/'prospective.json'),generator=pin(Path(__file__))))
    print(json.dumps(result))


if __name__=='__main__':main()

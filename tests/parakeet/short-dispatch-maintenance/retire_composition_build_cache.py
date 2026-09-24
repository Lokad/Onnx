"""Retire the closed V2 composition package cache, preserving every package archive."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/validated-composition-root-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
SOURCE=ROOT/'artifacts/parakeet-validated-composition-build-amd-v2-20260924'
OUT=ROOT/'artifacts/parakeet-composition-build-cache-retention-20260924'
REMOTE='/dev/shm/lokad-parakeet-validated-composition-build-v2-20260924'
FEED='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed'


def main():
    assert not OUT.exists()
    assert pin(SOURCE/'closed.json')['sha256']=='bb578d7ad6dfeaf989726476d6d7864438e602a41c7727a99ee05cb8c9eb71da'
    proof=read(SOURCE/'closed.json');assert proof['passed']
    for key,name in [('analysis','analysis.json'),('build_review','build-review.json'),
        ('collection','capture-collected/capture-collection.json'),('transfer','capture-transfer.json')]:
        assert proof[key]==pin(SOURCE/name)
    collections={};owners=[]
    for kind in ['build','capture']:
        folder=SOURCE/(kind+'-collected');name=kind+'-collection.json';receipt=read(folder/name)
        assert receipt['terminal'] and receipt['code']==0
        for relative,wanted in receipt['files'].items():assert pin(folder/relative)==wanted,relative
        state=read(folder/(kind+'-state.json'));assert state['complete'] and state['code']==0
        owners.extend([state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()])
        collections[name]=pin(folder/name)
    common=PRELUDE+f'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
root=Path({REMOTE!r});cache=root/'packages';feed=Path({FEED!r})
assert root.resolve()==root and root.parent==Path('/dev/shm')
assert cache.is_dir() and not cache.is_symlink()
assert not any(live(i) for i in {owners!r})
for name,wanted in {collections!r}.items():assert pin(root/name)==wanted
assert pin(root/'spec.json')=={pin(SOURCE/'bundle/spec.json')!r}
spec=read(root/'spec.json')
for name,wanted in spec['files'].items():assert pin(root/name)==wanted,name
for name,wanted in spec['external'].items():assert pin(name)==wanted,name
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
for p in cache.rglob('*'):
 assert not p.is_symlink() and p.resolve().is_relative_to(cache)
 if not p.is_file():continue
 assert str(p.resolve())==str(p) and str(p) not in protected
 files[str(p)]=pin(p)
 if p.suffix=='.nupkg':
  original=feed/p.name
  assert str(original.resolve()) in protected and pin(original)==files[str(p)]
  retained[str(original)]=pin(original)
assert files and retained
print(json.dumps(dict(files=files,retained=retained,protected_paths=len(protected))))
''',300))
    OUT.mkdir();save(OUT/'prospective.json',snapshot)
    result=json.loads(ssh(common+f'''
value={snapshot!r}
assert {{str(p) for p in cache.rglob('*') if p.is_file()}}==set(value['files'])
for name,wanted in value['files'].items():
 p=Path(name);assert not p.is_symlink() and p.resolve()==p and p.is_relative_to(cache)
 assert name not in protected and pin(p)==wanted,name
for name,wanted in value['retained'].items():assert name not in value['files'] and pin(name)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in value['files']:Path(name).unlink()
assert all(not Path(name).exists() for name in value['files'])
for name,wanted in value['retained'].items():assert pin(name)==wanted
for name,wanted in spec['files'].items():assert pin(root/name)==wanted,name
for name,wanted in spec['external'].items():assert pin(name)==wanted,name
idle();assert not any(live(i) for i in {owners!r})
print(json.dumps(dict(passed=True,files=len(value['files']),archives=len(value['retained']),
 logical_bytes=sum(v['bytes'] for v in value['files'].values()),protected_paths=len(protected),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    save(OUT/'closed.json',dict(**result,source_closure=pin(SOURCE/'closed.json'),
        prospective=pin(OUT/'prospective.json'),generator=pin(Path(__file__))))
    print(json.dumps(result))


if __name__=='__main__':main()

"""Share byte-identical binaries across terminal September 24 VM stages."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/observed-dense-where-numerics-amd'))
from run import ssh,PRELUDE
from protocol import pin,save
OUT=ROOT/'artifacts/parakeet-projection-preflight-runtime-links-20260924'


def main():
    assert not OUT.exists()
    snapshot=json.loads(ssh(PRELUDE+'''
from remote import idle,live
from protocol import pin,read
idle();assert psutil.boot_time()==1789634288.0
groups={};files={};owners=[];receipts={};excluded=[]
for root in sorted(Path('/dev/shm').glob('lokad-*20260924')):
 if root.is_symlink() or root.resolve()!=root:continue
 found=[];identities=[]
 for kind in ['','build-','capture-']:
  p=root/(kind+'collection.json')
  if not p.is_file():continue
  receipt=read(p)
  if not receipt['terminal'] or receipt.get('input_error') is not None:
   excluded.append(str(root));found=[];break
  ids=receipt.get('identities')
  if ids is None:
   state_path=root/(kind+'state.json');assert pin(state_path)==receipt['state']
   state=read(state_path);assert state['complete']
   ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
  assert not any(live(i) for i in ids)
  identities.extend(ids);found.append(p)
 if not found:continue
 owners.extend(identities);receipts.update({str(p):pin(p) for p in found})
 candidates=set()
 for sub in ['runtimes','runtime','measured','previous','tracer','runtime-observed','source/runtime-observed','built','bridge']:
  candidates.update((root/sub).rglob('*'))
 for folder in (root/'source').glob('**/bin'):
  candidates.update(folder.rglob('*'))
 for path in sorted(candidates):
  if not path.is_file() or path.suffix not in ['.dll','.pdb','.so']:continue
  assert not path.is_symlink() and path.resolve()==path and path.is_relative_to(root)
  identity=pin(path);stat=path.stat();key=(identity['sha256'],identity['bytes'],stat.st_mode,stat.st_uid,stat.st_gid)
  files[str(path)]=identity
  groups.setdefault(key,[]).append(dict(path=str(path),inode=stat.st_ino))
links=[]
for rows in groups.values():
 source=rows[0]
 for target in rows[1:]:
  if source['inode']!=target['inode']:links.append(dict(source=source['path'],target=target['path'],identity=files[target['path']]))
print(json.dumps(dict(passed=True,files=files,owners=owners,receipts=receipts,links=links,excluded=excluded)))
''',300))
    OUT.mkdir();save(OUT/'prospective.json',snapshot)
    result=json.loads(ssh(PRELUDE+f'''
from remote import idle,live
from protocol import pin
idle();assert psutil.boot_time()==1789634288.0
assert not any(live(i) for i in {snapshot['owners']!r})
files={snapshot['files']!r};links={snapshot['links']!r}
for name,wanted in {snapshot['receipts']!r}.items():assert pin(Path(name))==wanted,name
for name,wanted in files.items():assert pin(Path(name))==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in links:
 source=Path(row['source']);target=Path(row['target']);temporary=target.with_suffix(target.suffix+'.projection-preflight-link')
 assert source.is_relative_to('/dev/shm') and target.is_relative_to('/dev/shm')
 assert source.resolve()==source and target.resolve()==target and not temporary.exists()
 assert not source.is_symlink() and not target.is_symlink()
 a,b=source.stat(),target.stat();assert (a.st_mode,a.st_uid,a.st_gid)==(b.st_mode,b.st_uid,b.st_gid)
 assert pin(source)==pin(target)==row['identity']
 os.link(source,temporary);temporary.replace(target)
 assert source.stat().st_ino==target.stat().st_ino and pin(target)==row['identity']
for name,wanted in files.items():assert pin(Path(name))==wanted,name
assert not any(live(i) for i in {snapshot['owners']!r});idle()
print(json.dumps(dict(passed=True,links=len(links),verified_files=len(files),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    save(OUT/'closed.json',dict(**result,prospective=pin(OUT/'prospective.json'),generator=pin(Path(__file__))))
    print(json.dumps(result))


if __name__=='__main__':main()

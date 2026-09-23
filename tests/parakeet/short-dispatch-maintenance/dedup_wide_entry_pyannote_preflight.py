"""Reclaim identical closed VM tensors/binaries between numerical workers."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-pyannote-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-wide-entry-pyannote-preflight-maintenance-20260923'

def main():
    assert not BASE.exists();BASE.mkdir()
    pause=json.loads(ssh(PRELUDE+'''
from protocol import read,pin
from remote import live
owner=read(base/'deployment.json');assert owner==dict(pid=865729,birth=1790194929.96)
p=psutil.Process(owner['pid']);assert live(owner) and p.name().startswith('python')
assert str(base/'tools/remote.py') in p.cmdline()
state=read(base/'identity.json');assert not state['complete'] and all(r['complete'] for r in state['runs'])
assert [r['name'] for r in state['runs']]==['consumer-restore','consumer-build','consumer-inventory','selected']
assert not p.children(recursive=True)
pre=max(base.glob('*-preflight.json'),key=lambda p:p.stat().st_mtime)
assert pre.name=='candidate-preflight.json' and read(pre)[-1]['seconds']>=120
assert read(pre)[-1]['available']<12*1024**3
p.suspend()
state2=read(base/'identity.json')
if state2!=state or p.children(recursive=True):
 p.resume();raise AssertionError('Worker boundary moved; no maintenance')
print(json.dumps(dict(passed=True,owner=owner,state=pin(base/'identity.json'),preflight=pre.name,observations=read(pre),paused=time.time(),
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''))
    save(BASE/'paused.json',pause)
    snapshot=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read
from remote import live
owner={pause['owner']!r};p=psutil.Process(owner['pid'])
assert live(owner) and p.status()==psutil.STATUS_STOPPED and not p.children(recursive=True)
assert pin(base/'identity.json')=={pause['state']!r}
files={{}};groups={{}};owners=[];inodes={{}}
for root in sorted(Path('/dev/shm').glob('lokad-*')):
 if root==base or not root.is_dir() or root.is_symlink() or not (root/'collection.json').exists():continue
 receipt=read(root/'collection.json')
 if not receipt.get('terminal') or receipt.get('input_error') is not None:continue
 assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
 for path in root.rglob('*'):
  if not path.is_file() or path.suffix not in ['.dll','.pdb','.so','.npy','.f32']:continue
  assert not path.is_symlink() and path.resolve().is_relative_to(root)
  stat=path.stat();inode=(stat.st_dev,stat.st_ino)
  if inode not in inodes:inodes[inode]=pin(path)
  identity=inodes[inode];assert identity['bytes']==stat.st_size
  key=(identity['sha256'],identity['bytes'],stat.st_mode,stat.st_dev,stat.st_uid,stat.st_gid)
  files[str(path)]=identity
  groups.setdefault(key,[]).append(dict(path=str(path),inode=stat.st_ino,links=stat.st_nlink))
links=[]
for rows in groups.values():
 rows.sort(key=lambda r:(-r['links'],r['path']));source=rows[0]
 for target in rows[1:]:
  if source['inode']!=target['inode']:links.append(dict(source=source['path'],target=target['path'],identity=files[target['path']]))
print(json.dumps(dict(passed=True,files=files,owners=owners,links=links,unique_inodes=len(inodes))))
''',300))
    save(BASE/'prospective.json',snapshot)
    result=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read,verify
from remote import live
owner={pause['owner']!r};p=psutil.Process(owner['pid'])
assert live(owner) and p.status()==psutil.STATUS_STOPPED and not p.children(recursive=True)
assert pin(base/'identity.json')=={pause['state']!r}
assert not any(live(i) for i in {snapshot['owners']!r})
files={snapshot['files']!r};links={snapshot['links']!r}
verified={{}}
for name,wanted in files.items():
 path=Path(name);stat=path.stat();key=(stat.st_dev,stat.st_ino)
 if key not in verified:verified[key]=pin(path)
 assert verified[key]==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in links:
 source=Path(row['source']);target=Path(row['target']);temporary=target.with_suffix(target.suffix+'.m54-paused-dedup')
 assert source.resolve().is_relative_to('/dev/shm') and target.resolve().is_relative_to('/dev/shm')
 assert not source.is_relative_to(base) and not target.is_relative_to(base) and not temporary.exists()
 assert source.stat().st_mode==target.stat().st_mode and pin(source)==pin(target)==row['identity']
 assert (source.stat().st_uid,source.stat().st_gid)==(target.stat().st_uid,target.stat().st_gid)
 os.link(source,temporary);temporary.replace(target)
 assert source.stat().st_ino==target.stat().st_ino
verified={{}}
for name,wanted in files.items():
 path=Path(name);stat=path.stat();key=(stat.st_dev,stat.st_ino)
 if key not in verified:verified[key]=pin(path)
 assert verified[key]==wanted,name
verify(base)
after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
assert after['available']>=12*1024**3 and after['tmpfs']>=3*1024**3
assert live(owner) and p.status()==psutil.STATUS_STOPPED and pin(base/'identity.json')=={pause['state']!r}
p.resume()
print(json.dumps(dict(passed=True,links=len(links),verified_files=len(files),before=before,after=after,resumed=True,time=time.time())))
''',300))
    save(BASE/'closed.json',dict(**result,paused=pin(BASE/'paused.json'),prospective=pin(BASE/'prospective.json'),script=pin(__file__)))
    print(json.dumps(result))

if __name__=='__main__':main()

"""Deduplicate large immutable payload/evidence metadata in terminal VM campaigns."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-pyannote-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-closed-evidence-links-20260923'


def main():
    assert not BASE.exists();BASE.mkdir()
    snapshot=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read
from remote import live,idle
idle();assert psutil.boot_time()==1789634288.0
files={{}};groups={{}};owners=[];inodes={{}}
for root in sorted(Path('/dev/shm').glob('lokad-*')):
 if not root.is_dir() or root.is_symlink() or not (root/'collection.json').exists():continue
 receipt=read(root/'collection.json')
 if not receipt.get('terminal') or receipt.get('input_error') is not None:continue
 assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
 for path in root.rglob('*'):
  if not path.is_file() or path.suffix!='.json' or path.stat().st_size<1048576:continue
  relative=path.relative_to(root)
  if relative.parts[0]!='evidence' and relative.as_posix()!='payload.json':continue
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
from remote import live,idle
idle();assert psutil.boot_time()==1789634288.0
assert not any(live(i) for i in {snapshot['owners']!r})
files={snapshot['files']!r};links={snapshot['links']!r}
verified={{}}
for name,wanted in files.items():
 path=Path(name);stat=path.stat();key=(stat.st_dev,stat.st_ino)
 if key not in verified:verified[key]=pin(path)
 assert verified[key]==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in links:
 source=Path(row['source']);target=Path(row['target']);temporary=target.with_suffix(target.suffix+'.closed-evidence-dedup')
 assert source.resolve().is_relative_to('/dev/shm') and target.resolve().is_relative_to('/dev/shm')
 assert not temporary.exists()
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
idle()
print(json.dumps(dict(passed=True,links=len(links),verified_files=len(files),before=before,after=after,all_owners_terminal=True,time=time.time())))
''',300))
    save(BASE/'closed.json',dict(**result,prospective=pin(BASE/'prospective.json'),script=pin(__file__)))
    print(json.dumps(result))

if __name__=='__main__':main()

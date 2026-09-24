"""Share identical immutable binaries and large evidence files from closed M57 campaigns."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-pyannote-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-provider-where-postscreen-links-20260924'


def main():
    assert not BASE.exists();BASE.mkdir()
    snapshot=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read
from remote import live,idle
idle();assert psutil.boot_time()==1789634288.0
files={{}};groups={{}};owners=[];inodes={{}}
for root in [Path('/dev/shm')/name for name in [
 'lokad-parakeet-wide-entry-first-use-build-20260923',
 'lokad-parakeet-scalar-where-layout-20260923',
 'lokad-parakeet-scalar-where-build-v2-20260924',
 'lokad-parakeet-scalar-where-numerics-20260924',
 'lokad-parakeet-scalar-where-mask-bits-20260924',
 'lokad-parakeet-scalar-where-build-v3-20260924',
 'lokad-parakeet-scalar-where-numerics-v3-20260924',
 'lokad-parakeet-scalar-where-screen-20260924',
 'lokad-parakeet-scalar-where-fallback-codegen-20260924',
 'lokad-parakeet-provider-where-build-20260924',
 'lokad-parakeet-provider-where-numerics-20260924',
 'lokad-parakeet-provider-where-numerics-v2-20260924',
 'lokad-parakeet-provider-where-screen-20260924']]:
 assert root.is_dir() and not root.is_symlink() and (root/'collection.json').is_file()
 receipt=read(root/'collection.json')
 assert receipt['terminal'] and receipt['code']==(1 if root.name=='lokad-parakeet-provider-where-numerics-20260924' else 0) and receipt['input_error'] is None
 spec=read(root/'payload.json')
 for name,wanted in spec['files'].items():assert pin(root/name)==wanted,name
 assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
 for path in root.rglob('*'):
  if not path.is_file():continue
  relative=path.relative_to(root)
  metadata=path.suffix=='.json' and path.stat().st_size>=1048576 and (relative.parts[0]=='evidence' or relative.as_posix()=='payload.json')
  if path.suffix not in ['.dll','.pdb','.so'] and not metadata:continue
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
 source=Path(row['source']);target=Path(row['target']);temporary=target.with_suffix(target.suffix+'.m58-postscreen-dedup')
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

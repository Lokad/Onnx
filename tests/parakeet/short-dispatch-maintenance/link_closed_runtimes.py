"""Deduplicate identical immutable runtime binaries from terminal September 23 jobs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/first-use-kernels-screen'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-first-use-kernels-runtime-links-20260923'
assert not BASE.exists();BASE.mkdir()
snapshot=json.loads(ssh(PRELUDE+'''
from remote import idle,live
from protocol import pin,read
idle();assert psutil.boot_time()==1789634288.0
groups={};files={};owners=[]
for root in sorted(Path('/dev/shm').glob('lokad-*20260923')):
 if not root.name.startswith(('lokad-parakeet-','lokad-pyannote-winograd-')) or not (root/'collection.json').exists():continue
 receipt=read(root/'collection.json');assert receipt['terminal'] and receipt['input_error'] is None
 assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
 for sub in ['runtimes','runtime','measured','previous','tracer']:
  for path in sorted((root/sub).rglob('*')):
   if not path.is_file() or path.suffix not in ['.dll','.pdb','.so']:continue
   assert not path.is_symlink() and path.resolve().is_relative_to(root)
   identity=pin(path);stat=path.stat();key=(identity['sha256'],identity['bytes'],stat.st_mode)
   files[str(path)]=identity
   groups.setdefault(key,[]).append(dict(path=str(path),inode=stat.st_ino))
links=[]
for rows in groups.values():
 source=rows[0]
 for target in rows[1:]:
  if source['inode']!=target['inode']:links.append(dict(source=source['path'],target=target['path'],identity=files[target['path']]))
print(json.dumps(dict(passed=True,files=files,owners=owners,links=links)))
'''))
save(BASE/'prospective.json',snapshot)
result=json.loads(ssh(PRELUDE+f'''
from remote import idle,live
from protocol import pin
idle();assert psutil.boot_time()==1789634288.0
assert not any(live(i) for i in {snapshot['owners']!r})
files={snapshot['files']!r};links={snapshot['links']!r}
for name,wanted in files.items():assert pin(Path(name))==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in links:
 source=Path(row['source']);target=Path(row['target']);temporary=target.with_suffix(target.suffix+'.dedup-link')
 assert source.is_relative_to('/dev/shm') and target.is_relative_to('/dev/shm') and not temporary.exists()
 assert source.stat().st_mode==target.stat().st_mode and pin(source)==pin(target)==row['identity']
 os.link(source,temporary);temporary.replace(target)
 assert source.stat().st_ino==target.stat().st_ino and pin(target)==row['identity']
for name,wanted in files.items():assert pin(Path(name))==wanted,name
print(json.dumps(dict(passed=True,links=len(links),verified_files=len(files),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''))
save(BASE/'closed.json',dict(**result,prospective=pin(BASE/'prospective.json')))
print(json.dumps(result))

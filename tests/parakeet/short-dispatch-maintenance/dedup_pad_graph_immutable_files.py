"""Share identical immutable file storage while the graph supervisor is paused."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-current-graphs-v2-amd'))
from run import BASE,ssh,PRELUDE
from protocol import pin,read,save

OUT=ROOT/'artifacts/parakeet-pad-graph-immutable-file-links-20260926'


def main():
    assert not OUT.exists();owner=read(BASE/'deployment.json')
    assert owner==dict(pid=1126961,birth=1790456331.32)
    OUT.mkdir();save(OUT/'prospective.json',dict(owner=owner,payload=pin(BASE/'payload.json'),helper=pin(__file__)))
    result=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read,save,verify
from remote import live
expected={owner!r};assert pin(base/'payload.json')=={pin(BASE/'payload.json')!r}
state=read(base/'identity.json');assert state['supervisor']==expected and not state['complete']
assert state['runs'] and all(r['complete'] and r['code']==0 for r in state['runs'])
process=psutil.Process(expected['pid']);assert live(expected)
paused=False
try:
 process.suspend();paused=True;deadline=time.monotonic()+5
 while process.status()!=psutil.STATUS_STOPPED and time.monotonic()<deadline:time.sleep(.01)
 assert process.status()==psutil.STATUS_STOPPED and not process.children(recursive=True)
 assert read(base/'identity.json')==state
 ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in ancestors or p.pid==expected['pid']:continue
  assert p.info['name'] not in ['dotnet','perf']
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
 spec=verify(base);files={{}};groups={{}};inodes={{}};owners=[]
 for root in sorted(Path('/dev/shm').glob('lokad-*')):
  if not root.is_dir() or root.is_symlink():continue
  if root!=base:
   if not (root/'collection.json').is_file():continue
   receipt=read(root/'collection.json')
   if not receipt.get('terminal') or receipt.get('input_error') is not None:continue
   assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
  for path in root.rglob('*'):
   if not path.is_file() or path.stat().st_size<16384:continue
   relative=path.relative_to(root).as_posix()
   if root==base and relative not in spec['files']:continue
   assert not path.is_symlink() and path.resolve().is_relative_to(root)
   stat=path.stat();inode=(stat.st_dev,stat.st_ino)
   if inode not in inodes:inodes[inode]=pin(path)
   identity=inodes[inode];assert identity['bytes']==stat.st_size
   if root==base:assert identity==spec['files'][relative]
   key=(identity['sha256'],identity['bytes'],stat.st_mode,stat.st_dev,stat.st_uid,stat.st_gid)
   files[str(path)]=identity
   groups.setdefault(key,[]).append(dict(path=str(path),inode=stat.st_ino,links=stat.st_nlink))
 links=[]
 for rows in groups.values():
  rows.sort(key=lambda r:(-r['links'],r['path']));source=rows[0]
  for target in rows[1:]:
   if source['inode']!=target['inode']:links.append(dict(source=source['path'],target=target['path'],identity=files[target['path']]))
 assert links and read(base/'identity.json')==state and not process.children(recursive=True)
 ledger=Path('/dev/shm/lokad-pad-graph-immutable-file-links-20260926.json');assert not ledger.exists()
 save(ledger,dict(prospective=True,owner=expected,files=files,links=links,closed_owners=owners))
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
 for row in links:
  source=Path(row['source']);target=Path(row['target']);temporary=target.with_name(target.name+'.pad-graph-immutable-link')
  assert source.resolve().is_relative_to('/dev/shm') and target.resolve().is_relative_to('/dev/shm')
  assert not temporary.exists() and pin(source)==pin(target)==row['identity']
  assert source.stat().st_mode==target.stat().st_mode
  assert (source.stat().st_uid,source.stat().st_gid)==(target.stat().st_uid,target.stat().st_gid)
  os.link(source,temporary);temporary.replace(target)
  assert source.stat().st_ino==target.stat().st_ino
 verified={{}}
 for name,wanted in files.items():
  path=Path(name);stat=path.stat();key=(stat.st_dev,stat.st_ino)
  if key not in verified:verified[key]=pin(path)
  assert verified[key]==wanted,name
 verify(base);assert read(base/'identity.json')==state
 value=dict(passed=True,links=len(links),verified_files=len(files),before=before,
  after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
  owner=expected,no_inference_during_cleanup=True,all_bytes_preserved=True,
  files=files,operations=links,closed_owners=owners)
 save(ledger,value)
finally:
 if paused and live(expected) and process.status()==psutil.STATUS_STOPPED:process.resume()
assert live(expected) and process.status()!=psutil.STATUS_STOPPED
value['same_owner_resumed']=True
print(json.dumps(value))
''',300))
    save(OUT/'closed.json',dict(**result,prospective=pin(OUT/'prospective.json')))
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','operations','closed_owners']}))


if __name__=='__main__':main()

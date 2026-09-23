"""Retire terminal VM diagnostic copies already preserved in closed local archives."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'tests/benchmarks/startup-diagnostic-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/warmed-release-vm-retention-20260923'
assert not BASE.exists();BASE.mkdir();files={};receipts=[]
for name,remote,digest in [('graph-startup-diagnostic-amd-20260923','lokad-graph-startup-diagnostic-20260923','c2f4961396ec577711ef3ac48e629e901414a21a327594921b59fb015bda9a1f')]:
 folder=ROOT/'artifacts'/name;assert pin(folder/'closed.json')['sha256']==digest
 for relative,wanted in read(folder/'closed.json')['files'].items():assert pin(folder/relative)==wanted,relative
 receipt=read(folder/'collected/collection.json');assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
 receipts.append(dict(root='/dev/shm/'+remote,identity=pin(folder/'collected/collection.json'),owners=receipt['identities']))
 for relative,wanted in receipt['files'].items():
  if relative.endswith('/events/events.jsonl'):
   assert pin(folder/'collected'/relative)==wanted
   files['/dev/shm/'+remote+'/'+relative]=dict(identity=wanted,local=(folder/'collected'/relative).relative_to(ROOT).as_posix())
assert len(files)==2
save(BASE/'prospective.json',dict(passed=True,files=files,receipts=receipts))
result=json.loads(ssh(PRELUDE+f'''
from remote import idle,live
from protocol import pin,read
idle();assert psutil.boot_time()==1789634288.0
files={files!r};receipts={receipts!r}
for receipt in receipts:
 assert pin(Path(receipt['root'])/'collection.json')==receipt['identity']
 assert not any(live(i) for i in receipt['owners'])
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 if not (folder/'payload.json').exists():continue
 payload=read(folder/'payload.json')
 protected.update(str((folder/name).resolve()) for name in payload.get('files',{{}}))
 protected.update(payload.get('external',{{}}))
for name,record in files.items():
 target=Path(name).resolve();assert str(target)==name and str(target) not in protected
 assert any(target.is_relative_to(Path(r['root'])) for r in receipts) and pin(target)==record['identity'],name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in files:Path(name).unlink()
assert all(not Path(name).exists() for name in files)
print(json.dumps(dict(passed=True,files=len(files),bytes=sum(v['identity']['bytes'] for v in files.values()),before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''))
save(BASE/'closed.json',dict(**result,prospective=pin(BASE/'prospective.json')));print(json.dumps(result))

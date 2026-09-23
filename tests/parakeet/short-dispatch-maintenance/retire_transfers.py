"""Retire verified transfer copies and two locally restored terminal profile traces."""
import hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/short-dispatch-numerics-v2'))
import run
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-short-dispatch-transfer-maintenance-20260923'
assert not BASE.exists() or not any(BASE.iterdir());BASE.mkdir(exist_ok=True)
pairs=[('parakeet-current-profile-amd','parakeet-current-profile'),('pyannote-winograd-profile-amd','pyannote-winograd-profile'),('parakeet-current-profile-build-amd','parakeet-current-profile-build'),('pyannote-winograd-profile-build-amd','pyannote-winograd-profile-build'),('parakeet-wide-per-call-build-amd','parakeet-wide-per-call-build'),('parakeet-short-wide-pack-build-amd','parakeet-short-wide-pack-build'),('parakeet-short-dispatch-build-amd','parakeet-short-dispatch-build'),('parakeet-short-dispatch-build-amd-v2','parakeet-short-dispatch-build-v2'),('parakeet-short-dispatch-build-amd-v3','parakeet-short-dispatch-build-v3')]
targets={};owners={}
for local,remote in pairs:
 folder=ROOT/'artifacts'/(local+'-20260923');base='/dev/shm/lokad-'+remote+'-20260923'
 receipt=read(folder/'collected/collection.json');assert receipt['terminal'] and receipt['input_error'] is None
 owners[base]=dict(receipt=pin(folder/'collected/collection.json'),identities=receipt['identities'])
 content=folder/('bundle' if (folder/'bundle').is_dir() else 'payload')
 archive=folder/'payload.tar.gz';assert pin(archive)==read(folder/'prepared.json')['archive']
 with tarfile.open(archive) as tar:
  for m in tar:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   stream=tar.extractfile(m);digest=hashlib.file_digest(stream,'sha256').hexdigest()
   assert dict(bytes=m.size,sha256=digest)==pin(content/m.name)
 targets[base+'/transfer.tar.gz']=pin(archive)
folder=ROOT/'artifacts/parakeet-current-profile-amd-20260923'
with tarfile.open(folder/'results.tar.gz') as tar:
 for name in ['sampled-a/capture.nettrace','sampled-b/capture.nettrace']:
  expected=pin(folder/'collected'/name)
  assert expected==read(folder/'collected/collection.json')['files'][name]
  m=tar.getmember(name);stream=tar.extractfile(m)
  assert dict(bytes=m.size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())==expected
  targets['/dev/shm/lokad-parakeet-current-profile-20260923/'+name]=expected
save(BASE/'request.json',dict(targets=targets,owners=owners))
result=json.loads(run.ssh(run.PRELUDE+f"""
from protocol import pin,read
from remote import idle,live
idle()
owners={owners!r};targets={targets!r}
for name,row in owners.items():
 assert pin(Path(name)/'collection.json')==row['receipt']
 assert all(not live(i) for i in row['identities'])
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 if (folder/'payload.json').exists():
  payload=read(folder/'payload.json')
  protected.update(str((folder/n).resolve()) for n in payload.get('files',{{}}))
  protected.update(payload.get('external',{{}}))
for name,wanted in targets.items():
 p=Path(name).resolve();assert str(p)==name and str(p) not in protected and pin(p)==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in targets:Path(name).unlink()
print(json.dumps(dict(passed=True,files=len(targets),bytes=sum(p['bytes'] for p in targets.values()),before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
"""))
save(BASE/'closed.json',dict(**result,request=pin(BASE/'request.json')));print(json.dumps(result))

"""Archive and retire only generated files from four terminal build namespaces."""
import hashlib,json,subprocess,sys,tarfile,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/short-dispatch-numerics'))
import run
BASE=ROOT/'artifacts/parakeet-short-dispatch-maintenance-20260923'
NAMES=['lokad-parakeet-wide-per-call-build-20260923','lokad-parakeet-short-wide-pack-build-20260923','lokad-parakeet-short-dispatch-build-20260923','lokad-parakeet-short-dispatch-build-v2-20260923']
SUBS=['packages']+[f'source/src/{project}/{kind}' for project in ['Lokad.Onnx','Lokad.Onnx.Data','Lokad.Onnx.CLI'] for kind in ['bin','obj']]
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
if (BASE/'snapshot.json').exists():
 assert (BASE/'snapshot.json').exists() and (BASE/'generated.tar.gz').exists() and not (BASE/'closed.json').exists()
 snapshot=json.loads((BASE/'snapshot.json').read_text())
else:
 assert not BASE.exists() or not any(BASE.iterdir());BASE.mkdir(exist_ok=True)
 script=run.PRELUDE+textwrap.dedent(f"""
 import hashlib
 from protocol import read,pin
 from remote import idle,live
 idle();assert psutil.boot_time()==1789634288.0
 roots=[Path('/dev/shm')/n for n in {NAMES!r}]
 inputs={{}};files={{}};owners=[]
 for root in roots:
  receipt=read(root/'collection.json');assert receipt['terminal'] and receipt['input_error'] is None
  assert all(not live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
  payload=read(root/'payload.json')
  for name,wanted in payload['files'].items():assert pin(root/name)==wanted,name
  inputs.update({{str((root/name).resolve()):wanted for name,wanted in payload['files'].items()}})
  inputs.update(payload['external'])
  for sub in {SUBS!r}:
   target=(root/sub).resolve();assert target.is_relative_to(root.resolve()) and target.is_dir()
   for p in target.rglob('*'):
    if p.is_file():
     assert str(p.resolve()) not in inputs
     files[p.relative_to('/dev/shm').as_posix()]=pin(p)
 assert all(str(Path('/dev/shm')/name) not in inputs for name in files)
 print(json.dumps(dict(passed=True,files=files,owners=owners,available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))
 """)
 snapshot=json.loads(run.ssh(script));save(BASE/'snapshot.json',snapshot)
 export=run.PRELUDE+textwrap.dedent(f"""
 from protocol import pin
 from remote import idle,live
 idle();assert all(not live(i) for i in {snapshot['owners']!r})
 files={snapshot['files']!r}
 for name,wanted in files.items():assert pin(Path('/dev/shm')/name)==wanted,name
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
  for name in sorted(files):tar.add(Path('/dev/shm')/name,arcname=name,recursive=False)
 """)
 with (BASE/'generated.tar.gz').open('xb') as out,(BASE/'export.stderr').open('x') as err:
  result=subprocess.run(run.SSH+['python3','-B','-'],input=export.encode(),stdout=out,stderr=err,timeout=300,creationflags=subprocess.CREATE_NO_WINDOW)
 assert result.returncode==0
restore=Path('\\\\?\\'+str((BASE/'restore-longpaths').resolve()));restore.mkdir()
with tarfile.open(BASE/'generated.tar.gz') as tar:
 members=tar.getmembers();assert len(members)==len(snapshot['files'])
 assert set(m.name for m in members)==set(snapshot['files'])
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 tar.extractall(restore,filter='data')
for name,wanted in snapshot['files'].items():assert pin(restore/name)==wanted,name
save(BASE/'restored.json',dict(passed=True,archive=pin(BASE/'generated.tar.gz'),files=snapshot['files']))
retire=run.PRELUDE+'''
from protocol import read,pin
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
'''+textwrap.dedent(f"""
import shutil
files={snapshot['files']!r}
assert all(not live(i) for i in {snapshot['owners']!r})
# Verify every immutable payload across namespaces before checking dependencies.
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 if not (folder/'payload.json').exists():continue
 payload=read(folder/'payload.json')
 protected.update(str((folder/name).resolve()) for name in payload.get('files',{{}}))
 protected.update(payload.get('external',{{}}))
for name,wanted in files.items():
 target=(Path('/dev/shm')/name).resolve()
 assert str(target) not in protected and pin(target)==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in {NAMES!r}:
 root=Path('/dev/shm')/name
 for sub in {SUBS!r}:
  target=(root/sub).resolve();assert target.is_relative_to(root.resolve()) and target.is_dir()
  actual={{p.relative_to('/dev/shm').as_posix() for p in target.rglob('*') if p.is_file()}}
  expected={{n for n in files if n.startswith(name+'/'+sub+'/')}}
  assert actual==expected
  shutil.rmtree(target)
print(json.dumps(dict(passed=True,files=len(files),before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
""")
receipt=json.loads(run.ssh(retire));save(BASE/'closed.json',dict(**receipt,archive=pin(BASE/'generated.tar.gz'),restored=pin(BASE/'restored.json')))
print(json.dumps(receipt))

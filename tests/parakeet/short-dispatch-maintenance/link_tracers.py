"""Share byte-identical immutable tracer files; preserve every path and byte."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/dispatch-events-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save
BASE=ROOT/'artifacts/parakeet-dispatch-tracer-links-20260923'
SOURCE=ROOT/'artifacts/parakeet-current-profile-amd-20260923/payload'
assert not BASE.exists() or not any(BASE.iterdir());BASE.mkdir(exist_ok=True)
manifest=read(SOURCE/'payload.json');files={}
for p in (SOURCE/'tracer').rglob('*'):
 if p.is_file():
  relative=p.relative_to(SOURCE).as_posix();assert pin(p)==manifest['files'][relative];files[p.relative_to(SOURCE/'tracer').as_posix()]=pin(p)
assert len(files)==53
save(BASE/'prospective.json',dict(passed=True,files=files,donor='/dev/shm/lokad-parakeet-current-profile-20260923/tracer',targets=['lokad-parakeet-dispatch-events-20260923','lokad-pyannote-current-profile-v2-20260922','lokad-pyannote-prepared-profile-v2-20260922','lokad-pyannote-winograd-profile-20260923']))
script=PRELUDE+f'''
from remote import idle
from protocol import pin,read
idle();assert psutil.boot_time()==1789634288.0
donor=Path('/dev/shm/lokad-parakeet-current-profile-20260923/tracer');files={files!r}
targets=['lokad-parakeet-dispatch-events-20260923','lokad-pyannote-current-profile-v2-20260922','lokad-pyannote-prepared-profile-v2-20260922','lokad-pyannote-winograd-profile-20260923']
for name,wanted in files.items():assert pin(donor/name)==wanted,name
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
rows=[]
for namespace in targets:
 root=Path('/dev/shm')/namespace;tracer=root/'tracer'
 assert {{p.relative_to(tracer).as_posix() for p in tracer.rglob('*') if p.is_file()}}==set(files)
 payload=read(root/'payload.json') if (root/'payload.json').exists() else read(root/'stage.json')
 for name,wanted in files.items():
  target=(tracer/name).resolve();source=(donor/name).resolve()
  assert target.is_relative_to(tracer.resolve()) and source.is_relative_to(donor.resolve())
  assert not (tracer/name).is_symlink() and not (donor/name).is_symlink()
  assert pin(target)==wanted==payload['files']['tracer/'+name]
  assert target.stat().st_dev==source.stat().st_dev
  rows.append(dict(target=str(target),source=str(source),before_inode=target.stat().st_ino,source_inode=source.stat().st_ino,identity=wanted))
for row in rows:
 target=Path(row['target']);source=Path(row['source']);assert pin(target)==pin(source)==row['identity']
 if target.stat().st_ino!=source.stat().st_ino:
  temporary=target.with_name(target.name+'.shared-link');assert not temporary.exists()
  os.link(source,temporary);temporary.replace(target)
 assert pin(target)==row['identity'] and target.stat().st_ino==source.stat().st_ino
print(json.dumps(dict(passed=True,files=rows,before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''
value=json.loads(ssh(script));save(BASE/'closed.json',value)
print(json.dumps(dict(passed=True,files=len(value['files']),before=value['before'],after=value['after'],closed=pin(BASE/'closed.json'))))

"""Retire four closed VM trace duplicates after full local collection and event audit."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-warmed-qualification-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save

SOURCE=ROOT/'artifacts/e5-runtime-diagnostic-amd-20260924'
OUT=ROOT/'artifacts/e5-diagnostic-trace-retention-20260924'
REMOTE='/dev/shm/lokad-e5-runtime-diagnostic-20260924'
SUPERVISOR=dict(pid=957541,birth=1790266058.08)


def main():
    assert not OUT.exists()
    proof=read(SOURCE/'closed.json')
    assert proof['passed'] and proof['diagnostic_only']
    assert pin(SOURCE/'closed.json')['sha256']=='8304dc71470c5666b83078f1fa487ae8595259d3ce9738471f70065d4919e990'
    for name,wanted in proof['files'].items():assert pin(SOURCE/name)==wanted,name
    targets={f'{r}-capture/capture.nettrace':pin(SOURCE/f'collected/{r}-capture/capture.nettrace') for r in ['a','b','c','d']}
    receipt=read(SOURCE/'collected/collection.json')
    script=PRELUDE+f'''
from protocol import pin,read
from remote import live
assert psutil.boot_time()==1789634288.0
def between_workers():
 state=read(base/'identity.json');assert state['supervisor']=={SUPERVISOR!r}
 assert all(r['complete'] and r['code']==0 for r in state['runs'])
 assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
 own=psutil.Process();allowed={{own.pid,*[p.pid for p in own.parents()],{SUPERVISOR['pid']}}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in allowed:continue
  command=' '.join(p.info['cmdline'] or [])
  assert p.info['name'] not in ['dotnet','perf'],p.info
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command),p.info
 return dict(supervisor_live=live(state['supervisor']),completed=len(state['runs']),complete=state['complete'])
first=between_workers()
root=Path({REMOTE!r});assert root.resolve().parent==Path('/dev/shm')
assert not any(live(i) for i in {receipt['identities']!r})
assert pin(root/'collection.json')=={pin(SOURCE/'collected/collection.json')!r}
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  if not (folder/name).exists():continue
  value=read(folder/name)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{{}}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{{}}))
targets={targets!r}
for name,wanted in targets.items():
 target=root/name;assert target.resolve().is_relative_to(root.resolve()) and target.resolve()==target
 assert not target.is_symlink() and target.stat().st_nlink==1
 assert str(target) not in protected and pin(target)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
last=between_workers();assert last==first
for name in targets:(root/name).unlink()
assert all(not (root/name).exists() for name in targets)
print(json.dumps(dict(passed=True,retired=targets,bytes_retired=sum(v['bytes'] for v in targets.values()),
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 protected_paths=len(protected),between_workers=last)))
'''
    result=json.loads(ssh(script));assert result['passed']
    for name,wanted in targets.items():assert pin(SOURCE/'collected'/name)==wanted
    OUT.mkdir();save(OUT/'closed.json',dict(**result,local_source=SOURCE.relative_to(ROOT).as_posix(),
        source_closure=pin(SOURCE/'closed.json'),generator=pin(Path(__file__))))
    print(json.dumps(result))


if __name__=='__main__':main()

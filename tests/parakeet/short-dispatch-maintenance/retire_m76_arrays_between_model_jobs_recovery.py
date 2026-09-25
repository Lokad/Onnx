"""Retire closed M76 tensor duplicates while the M78 correctness owner is paused."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/packed-final-row-models-amd'))
from run import ssh,PRELUDE,BASE as ACTIVE,pin,read,save

OUT=ROOT/'artifacts/parakeet-m78-model-headroom-recovery-20260925'
OLD=ROOT/'artifacts/parakeet-owned-packed-weight-models-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-models-20260925'


def main():
    assert not OUT.exists()
    assert pin(OLD/'closed.json')['sha256']=='bb1a758c6936fa251d921992c67dd7c65ff075fe20829af72f4455b529f1b2ca'
    closure=read(OLD/'closed.json');receipt=read(OLD/'collected/collection.json')
    assert closure['passed'] and closure['analysis']==pin(OLD/'analysis.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert closure['files']['collected/collection.json']==pin(OLD/'collected/collection.json')
    jobs=read(OLD/'payload.json')['jobs']
    names=sorted(n for n in receipt['files'] if any(n.startswith(j+'/') for j in jobs) and Path(n).suffix in ['.bin','.f32'])
    assert len(names)==3136
    targets=[]
    for name in names:
        wanted=receipt['files'][name]
        assert pin(OLD/'collected'/name)==wanted==closure['files']['collected/'+name]
        targets.append(dict(path=REMOTE+'/'+name,local=(OLD/'collected'/name).relative_to(ROOT).as_posix(),identity=wanted))
    owner=read(ACTIVE/'deployment.json')
    assert owner==dict(pid=1039185,birth=1790328136.44)
    OUT.mkdir()
    prospective=dict(targets=targets,old_collection=pin(OLD/'collected/collection.json'),old_owners=receipt['identities'],
        owner=owner,active_payload=pin(ACTIVE/'payload.json'),helper=pin(Path(__file__)))
    save(OUT/'prospective.json',prospective)
    result=json.loads(ssh(PRELUDE+f'''
from protocol import pin,read
from remote import live
value={prospective!r};root=Path({REMOTE!r})
assert psutil.boot_time()==1789634288.0 and root.resolve()==root and root.parent==Path('/dev/shm')
assert pin(root/'collection.json')==value['old_collection'] and not any(live(i) for i in value['old_owners'])
assert pin(base/'payload.json')==value['active_payload'] and live(value['owner'])
payload=read(base/'payload.json');state=read(base/'identity.json')
assert not state['complete'] and state['supervisor']==value['owner']
assert state['runs'] and all(r['complete'] and r['code']==0 for r in state['runs'])
assert len(state['runs'])<len(payload['jobs'])
next_job=payload['jobs'][len(state['runs'])]
wait_path=base/(next_job+'-preflight.json');waiting=read(wait_path)
assert waiting and waiting[-1]['available']<payload['limits']['preflight_available']
paths=set();manifests={{}}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  p=folder/name
  if not p.is_file():continue
  record=read(p);manifests[str(p)]=pin(p)
  paths.update(str(folder/n) for n in record.get('files',{{}}))
  paths.update(record.get('external',{{}}))
protected={{str(Path(n).resolve()) for n in paths}}
for row in value['targets']:
 p=Path(row['path'])
 assert p.resolve()==p and p.is_relative_to(root) and not p.is_symlink()
 assert str(p) not in protected and p.stat().st_nlink==1 and pin(p)==row['identity']
# Suspend only the known correctness supervisor; immediately undo a launch race.
owner=psutil.Process(value['owner']['pid']);assert owner.create_time()==value['owner']['birth']
paused=time.time();owner.suspend()
try:
 deadline=time.monotonic()+2
 while owner.status()!=psutil.STATUS_STOPPED and time.monotonic()<deadline:time.sleep(.01)
 assert owner.status()==psutil.STATUS_STOPPED and not owner.children(recursive=True)
 state=read(base/'identity.json')
 assert state['supervisor']==value['owner'] and not state['complete']
 assert [r['name'] for r in state['runs']]==payload['jobs'][:len(state['runs'])]
 assert state['runs'] and all(r['complete'] and r['code']==0 for r in state['runs'])
 assert payload['jobs'][len(state['runs'])]==next_job
 assert not any(live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
 ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in ancestors or p.pid==owner.pid:continue
  command=' '.join(p.info['cmdline'] or [])
  assert p.info['name'] not in ['dotnet','perf']
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in command)
 assert pin(base/'payload.json')==value['active_payload']
 for name,wanted in manifests.items():assert pin(name)==wanted,name
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
 physical=0
 for row in value['targets']:
  p=Path(row['path']);assert p.resolve()==p and p.is_relative_to(root) and not p.is_symlink()
  assert p.stat().st_nlink==1 and pin(p)==row['identity'];physical+=p.stat().st_blocks*512
 for row in value['targets']:Path(row['path']).unlink()
 assert all(not Path(row['path']).exists() for row in value['targets'])
 assert pin(base/'payload.json')==value['active_payload']
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
 result=dict(passed=True,owner=value['owner'],next_job=next_job,completed=[r['name'] for r in state['runs']],
  no_model_worker_during_retirement=True,files=len(value['targets']),logical_bytes=sum(r['identity']['bytes'] for r in value['targets']),
  physical_bytes=physical,before=before,after=after,protected_manifests=manifests,paused=paused)
finally:
 if live(value['owner']):owner.resume()
deadline=time.monotonic()+2
while live(value['owner']) and owner.status()==psutil.STATUS_STOPPED and time.monotonic()<deadline:time.sleep(.01)
assert live(value['owner']) and owner.status()!=psutil.STATUS_STOPPED
result.update(resumed=time.time(),resumed_same_owner=True,pause_seconds=time.time()-paused)
print(json.dumps(result))
''',300))
    for row in targets:assert pin(ROOT/row['local'])==row['identity']
    save(OUT/'closed.json',dict(**result,prospective=pin(OUT/'prospective.json'),all_local_originals_retained=True,
        scope='3136 closed M76 VM tensor duplicates only; M78 correctness supervisor paused between workers, then resumed unchanged. No scored campaign or product changed.'))
    print(json.dumps({k:v for k,v in result.items() if k!='protected_manifests'}))


if __name__=='__main__':main()

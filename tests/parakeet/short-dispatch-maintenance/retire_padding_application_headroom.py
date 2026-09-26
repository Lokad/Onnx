"""Retire exact collected outputs and two closed build caches between workers."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-application-diagnostic-amd'))
from run import ssh, PRELUDE, BASE as ACTIVE
from protocol import read, pin, save

OUT = ROOT/'artifacts/parakeet-padding-application-headroom-20260926'
TARGETS = [
    ('parakeet-owned-batch-isolation-root-amd-20260925','parakeet-owned-batch-isolation-root-20260925','collected/collection.json','cache'),
    ('parakeet-owned-batch-isolation-root-policy-amd-20260925','parakeet-owned-batch-isolation-root-policy-20260925','collected/collection.json','cache'),
    ('parakeet-owned-batch-isolation-profile-resume-amd-20260925','parakeet-owned-batch-isolation-profile-20260925','collected/resume-collection.json','profile'),
    ('parakeet-packed-final-row-profile-closure-amd-20260925','parakeet-packed-final-row-profile-20260925','collected/closure-collection.json','profile'),
    ('parakeet-owned-batch-isolation-pyannote-amd-20260925','parakeet-owned-batch-isolation-pyannote-20260925','collected/collection.json','arrays'),
    ('parakeet-owned-batch-isolation-graphs-amd-20260925','parakeet-owned-batch-isolation-graphs-20260925','collected/collection.json','arrays'),
]


def main():
    assert not OUT.exists()
    roots = {}
    for local, remote, receipt_name, kind in TARGETS:
        folder = ROOT/'artifacts'/local
        closure = read(folder/'closed.json');receipt = read(folder/receipt_name)
        assert closure.get('passed') or closure.get('preserved_failure')
        assert receipt['terminal']
        assert pin(folder/receipt_name) in [v for v in closure.values() if isinstance(v,dict)] or closure.get('files',{}).get(receipt_name) == pin(folder/receipt_name)
        copies = {n:v for n,v in receipt['files'].items() if
            (kind == 'profile' and n.split('/')[0] in ['wall','phase','control']) or
            (kind == 'arrays' and Path(n).suffix in ['.bin','.f32'] and not n.startswith(('assets/','inputs/','reference/')))}
        for name,wanted in copies.items(): assert pin(folder/Path(receipt_name).parent/name) == wanted
        roots['/dev/shm/lokad-'+remote] = dict(kind=kind, copies=copies, local=str(folder/Path(receipt_name).parent),
            owners=closure.get('terminal_owners',receipt.get('identities',[])),closure=pin(folder/'closed.json'))
        assert roots['/dev/shm/lokad-'+remote]['owners']
    owner = read(ACTIVE/'deployment.json');assert owner == dict(pid=1111767,birth=1790442351.02)
    OUT.mkdir();save(OUT/'prospective.json',dict(roots=roots,owner=owner,active_payload=pin(ACTIVE/'payload.json'),helper=pin(__file__)))
    result = json.loads(ssh(PRELUDE+f'''
from protocol import read,pin,save
from remote import live
roots={roots!r};expected_owner={owner!r}
assert psutil.boot_time()==1789634288.0 and pin(base/'payload.json')=={pin(ACTIVE/'payload.json')!r}
state=read(base/'identity.json');payload=read(base/'payload.json')
assert state['supervisor']==expected_owner and not state['complete']
assert state['runs'] and all(r['complete'] and r['code']==0 for r in state['runs'])
assert payload['jobs'][len(state['runs'])]=='candidate-capture'
assert read(base/'candidate-capture-preflight.json')[-1]['available']<payload['limits']['preflight_available']
paths=set();manifests={{}}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  path=folder/name
  if not path.is_file():continue
  value=read(path);manifests[str(path)]=pin(path)
  paths.update(str(folder/n) for n in value.get('files',{{}}));paths.update(value.get('external',{{}}))
  for link in value.get('links',{{}}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):paths.add(link['source'])
protected={{str(Path(n).resolve()) for n in paths}}
targets={{}};preserved=[]
for name,info in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert not any(live(i) for i in info['owners'])
 candidates=[root/n for n in info['copies'] if (root/n).is_file()]
 if info['kind']=='cache':
  for parent in (root/'source').rglob('*'):
   if parent.is_dir() and parent.name in ['bin','obj'] and not any(x in ['bin','obj'] for x in parent.relative_to(root/'source').parts[:-1]):
    candidates.extend(p for p in parent.rglob('*') if p.is_file())
 for path in candidates:
  assert path.resolve()==path and not path.is_symlink() and path.is_relative_to(root)
  if str(path) in protected or path.stat().st_nlink!=1:
   preserved.append(str(path));continue
  identity=pin(path);relative=path.relative_to(root).as_posix()
  if info['kind']!='cache':assert identity==info['copies'][relative]
  else:assert path.is_relative_to(root/'source') and any(x in ['bin','obj'] for x in path.relative_to(root/'source').parts[:-1])
  targets[str(path)]=identity
assert targets
owner=psutil.Process(expected_owner['pid']);assert owner.create_time()==expected_owner['birth']
owner.suspend();paused=time.time()
try:
 assert owner.status()==psutil.STATUS_STOPPED and not owner.children(recursive=True)
 state=read(base/'identity.json')
 assert not state['complete'] and state['supervisor']==expected_owner
 assert all(r['complete'] and r['code']==0 for r in state['runs']) and payload['jobs'][len(state['runs'])]=='candidate-capture'
 ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in ancestors or p.pid==owner.pid:continue
  assert p.info['name'] not in ['dotnet','perf']
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
 for name,wanted in manifests.items():assert pin(name)==wanted
 physical=0
 for name,wanted in targets.items():
  p=Path(name);assert p.resolve()==p and not p.is_symlink() and p.stat().st_nlink==1 and name not in protected and pin(p)==wanted
  physical+=p.stat().st_blocks*512
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
 save(base/'headroom-retirement-prospective.json',dict(targets=targets,manifests=manifests,paused=paused))
 for name in targets:Path(name).unlink()
 assert all(not Path(name).exists() for name in targets)
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
 result=dict(passed=True,targets=targets,preserved=preserved,physical_bytes=physical,before=before,after=after,
             owner=expected_owner,paused=paused,closed_owners_verified=True,no_runtime_during_retirement=True)
 save(base/'headroom-retirement.json',result)
finally:
 if live(expected_owner) and owner.status()==psutil.STATUS_STOPPED:owner.resume()
assert live(expected_owner) and owner.status()!=psutil.STATUS_STOPPED
result.update(resumed=time.time(),resumed_same_owner=True)
print(json.dumps(result))
''',300))
    for info in roots.values():
        for name,wanted in info['copies'].items(): assert pin(Path(info['local'])/name) == wanted
    save(OUT/'closed.json',dict(**result,prospective=pin(OUT/'prospective.json'),all_local_outputs_retained=True))
    print(json.dumps({k:v for k,v in result.items() if k not in ['targets','preserved']}))


if __name__=='__main__':main()

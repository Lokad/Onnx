"""Retire locally retained model-output duplicates while no inference is running."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-current-shared-amd'))
import run
from protocol import pin,read,save

ACTIVE=run.BASE
OUT=ROOT/'artifacts/parakeet-pad-shared-headroom-retention-20260926'
TARGETS=[
    ('parakeet-pad-current-models-amd-20260926','parakeet-pad-current-models-20260926',
     '194eb9a48b64df22d19b51adfc3ebe3accc56004544620123c128c2a76f066bf'),
    ('parakeet-owned-batch-isolation-models-amd-20260925','parakeet-owned-batch-isolation-models-20260925',
     '1997eeb782df89975f4893820bc0fc246dec60d1839b2c63a86ad2bba1aecd9d')]


def main():
    assert not OUT.exists()
    roots={}
    for local,remote,digest in TARGETS:
        folder=ROOT/'artifacts'/local
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');receipt=read(folder/'collected/collection.json')
        assert proof['passed'] and proof['files']['collected/collection.json']==pin(folder/'collected/collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        copies={n:v for n,v in receipt['files'].items() if n.split('/')[0] in
            ['selected-native-512','selected-native-256','candidate-native-512','candidate-native-256'] and Path(n).suffix=='.bin'}
        assert len(copies)==3136 and sum(v['bytes'] for v in copies.values())==49448096
        for n,wanted in copies.items():assert pin(folder/'collected'/n)==wanted
        roots['/dev/shm/lokad-'+remote]=dict(local=local,copies=copies,owners=receipt['identities'],
            closure=pin(folder/'closed.json'),receipt=pin(folder/'collected/collection.json'))
    owner=read(ACTIVE/'deployment.json');assert owner==dict(pid=1123419,birth=1790453340.94)
    OUT.mkdir();save(OUT/'prospective.json',dict(roots=roots,owner=owner,payload=pin(ACTIVE/'payload.json'),helper=pin(__file__)))
    value=json.loads(run.transport.ssh(run.transport.PRELUDE+f'''
from protocol import pin,read,save,verify
from remote import live
roots={roots!r};expected={owner!r}
assert psutil.boot_time()==1789634288.0 and pin(base/'payload.json')=={pin(ACTIVE/'payload.json')!r}
state=read(base/'identity.json');assert state['supervisor']==expected
assert state['runs'] and all(r['complete'] and r['code']==0 for r in state['runs'])
paused=False;process=None
try:
 if live(expected):
  assert not state['complete']
  process=psutil.Process(expected['pid']);assert process.create_time()==expected['birth']
  process.suspend();paused=True;deadline=time.monotonic()+5
  while process.status()!=psutil.STATUS_STOPPED and time.monotonic()<deadline:time.sleep(.01)
  assert process.status()==psutil.STATUS_STOPPED and not process.children(recursive=True)
  assert read(base/'identity.json')==state
 else:assert state['complete'] and state['code']==0
 assert not any(live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
 ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
 for p in psutil.process_iter(['name','cmdline']):
  if p.pid in ancestors or (paused and p.pid==expected['pid']):continue
  assert p.info['name'] not in ['dotnet','perf']
  assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
 paths=set();manifests={{}}
 for folder in Path('/dev/shm').glob('lokad-*'):
  for name in ['payload.json','stage.json','spec.json']:
   p=folder/name
   if not p.is_file():continue
   v=read(p);manifests[str(p)]=pin(p)
   paths.update(str(folder/n) for n in v.get('files',{{}}));paths.update(v.get('external',{{}}))
   for link in v.get('links',{{}}).values():
    if isinstance(link,dict) and isinstance(link.get('source'),str):paths.add(link['source'])
 protected={{str(Path(p).resolve()) for p in paths}}
 selected={{}};preserved=[]
 for name,info in roots.items():
  root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
  assert pin(root/'collection.json')==info['receipt'] and not any(live(i) for i in info['owners'])
  for relative,wanted in info['copies'].items():
   p=root/relative
   if not p.exists() or p.is_symlink() or str(p.resolve()) in protected or p.stat().st_nlink!=1:
    preserved.append(str(p));continue
   assert p.resolve()==p and p.is_relative_to(root) and pin(p)==wanted
   selected[str(p)]=dict(**wanted,physical=p.stat().st_blocks*512)
 assert selected
 for name,wanted in manifests.items():assert pin(name)==wanted
 assert read(base/'identity.json')==state
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
 save(base/'tensor-retirement-prospective.json',dict(targets=selected,manifests=manifests,owner=expected,paused=paused))
 for name in selected:Path(name).unlink()
 assert all(not Path(name).exists() for name in selected)
 verify(base)
 for name,wanted in manifests.items():assert pin(name)==wanted
 value=dict(passed=True,files=len(selected),physical_bytes=sum(v['physical'] for v in selected.values()),
     before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
     owner=expected,paused_between_workers=paused,protected_inputs_preserved=True,closed_owners_verified=True,
     no_inference_during_cleanup=True,preserved=preserved,targets=selected)
 save(base/'tensor-retirement.json',value)
finally:
 if paused and live(expected) and process.status()==psutil.STATUS_STOPPED:process.resume()
if paused:assert live(expected) and process.status()!=psutil.STATUS_STOPPED
value['same_owner_resumed']=paused
print(json.dumps(value))
''',300))
    for info in roots.values():
        for n,wanted in info['copies'].items():assert pin(ROOT/'artifacts'/info['local']/'collected'/n)==wanted
    save(OUT/'closed.json',dict(**value,prospective=pin(OUT/'prospective.json'),all_local_bytes_retained=True))
    print(json.dumps({k:v for k,v in value.items() if k not in ['targets','preserved']}))


if __name__=='__main__':main()

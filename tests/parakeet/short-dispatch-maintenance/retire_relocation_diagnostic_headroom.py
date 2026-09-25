"""Retire closed VM trace duplicates and generated caches, preserving all evidence."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/benchmarks/e5-relocation-tier-diagnostic-amd'))
from run import ssh, PRELUDE
from protocol import pin, read, save

OUT = ROOT/'artifacts/e5-relocation-tier-headroom-retention-20260925'
NEXT = ROOT/'artifacts/e5-relocation-tier-diagnostic-amd-20260925'
TARGETS = [
    ('e5-direct-code-diagnostic-amd-20260925', 'lokad-e5-direct-code-diagnostic-20260925', 'traces',
     'de164f33d9d75d089905a54d95e6b08971d6644745d4959017f689079682bd92'),
    ('e5-direct-tier-diagnostic-v2-amd-20260925', 'lokad-e5-direct-tier-diagnostic-v2-20260925', 'traces',
     'e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc'),
    ('e5-repeatability-diagnostic-amd-20260925', 'lokad-e5-repeatability-diagnostic-20260925', 'traces',
     'fe8ec805d18864a3a8d8b5a052543a3eef466535adc2769efc8c6c6747338a7b'),
    ('parakeet-owned-batch-isolation-build-amd-20260925', 'lokad-parakeet-owned-batch-isolation-build-20260925', 'packages',
     '5dd53e90d4924a36bb6f43cc80fa939e9236492949542dcf659dc056d4ea3860'),
    ('parakeet-direct-depthwise-build-v2-amd-20260925', 'lokad-parakeet-direct-depthwise-build-v2-20260925', 'packages',
     '26e4da0a66bf37aeda09dd5b0a5144817585ab7311cf87f85c4bb9ded894d171'),
]


def main():
    assert not OUT.exists()
    roots = {}
    for local, remote, kind, digest in TARGETS:
        folder = ROOT/'artifacts'/local
        assert pin(folder/'closed.json')['sha256'] == digest
        closed = read(folder/'closed.json'); assert closed['passed']
        for name, wanted in closed['files'].items(): assert pin(folder/name) == wanted, name
        collection = 'capture-collected/capture-collection.json' if kind == 'packages' else 'collected/collection.json'
        receipt = read(folder/collection); assert receipt['terminal'] and receipt['code'] == 0
        copies = {name:wanted for name,wanted in receipt['files'].items()
                  if kind == 'traces' and name.split('/')[0] in [r+'-'+action for r in 'abcd' for action in ['capture','export']]}
        for name, wanted in copies.items(): assert pin(folder/'collected'/name) == wanted
        roots['/dev/shm/'+remote] = dict(local=local, kind=kind, collection_name=Path(collection).name,
            collection=pin(folder/collection), owners=receipt['identities'], copies=copies,
            source_closure=pin(folder/'closed.json'))
    stage = read(NEXT/'bundle/stage.json')
    future = {value['source']:value['identity'] for value in stage['links'].values()}
    common = PRELUDE.replace("base=Path('/dev/shm/lokad-e5-relocation-tier-diagnostic-20260925')",
                            "base=Path('/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925')")+'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
roots=__ROOTS__;future=__FUTURE__
protected=set(future);manifests={}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  path=folder/name
  if not path.is_file():continue
  value=read(path);manifests[str(path)]=pin(path)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{}))
  for link in value.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):protected.add(str(Path(link['source']).resolve()))
for name,wanted in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/wanted['collection_name'])==wanted['collection']
 receipt=read(root/wanted['collection_name']);assert receipt['terminal'] and receipt['code']==0
 assert not any(live(identity) for identity in wanted['owners'])
for name,wanted in future.items():assert pin(name)==wanted,name
'''
    common = common.replace('__ROOTS__', repr(roots)).replace('__FUTURE__', repr(future))
    snapshot = json.loads(ssh(common+'''
files={};retained={};physical=0;preserved=[]
for name,wanted in roots.items():
 root=Path(name)
 paths=[p for p in (root/'packages').rglob('*') if p.is_file()] if wanted['kind']=='packages' else [root/n for n in wanted['copies'] if (root/n).is_file()]
 for path in paths:
  assert path.resolve()==path and not path.is_symlink() and path.is_relative_to(root)
  if str(path) in protected or path.stat().st_nlink!=1:
   preserved.append(str(path));continue
  assert path.stat().st_nlink==1,path
  identity=pin(path)
  if wanted['kind']=='traces':assert identity==wanted['copies'][path.relative_to(root).as_posix()]
  elif path.suffix=='.nupkg':
   original=Path(read(root/'spec.json')['feed'])/path.name
   assert original.resolve()==original and str(original) in protected and pin(original)==identity
   retained[str(original)]=identity
  files[str(path)]=identity;physical+=path.stat().st_blocks*512
assert files and retained
print(json.dumps(dict(passed=True,files=files,retained=retained,manifests=manifests,
 protected_paths=len(protected),preserved=preserved,physical_bytes=physical,
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''', 300))
    OUT.mkdir(); save(OUT/'prospective.json', dict(**snapshot, roots=roots, next_stage=pin(NEXT/'bundle/stage.json')))
    result = json.loads(ssh(common+'''
value=__SNAPSHOT__
assert manifests==value['manifests']
for name,wanted in value['files'].items():
 path=Path(name)
 assert path.resolve()==path and not path.is_symlink() and path.stat().st_nlink==1 and name not in protected
 assert any(path.is_relative_to(Path(root)/'packages') if info['kind']=='packages' else path.relative_to(Path(root)).as_posix() in info['copies']
            for root,info in roots.items() if path.is_relative_to(Path(root)))
 assert pin(path)==wanted,name
for name,wanted in value['retained'].items():assert name not in value['files'] and pin(name)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in value['files']:Path(name).unlink()
assert all(not Path(name).exists() for name in value['files'])
for name,wanted in {**future,**value['retained']}.items():assert pin(name)==wanted,name
idle()
print(json.dumps(dict(passed=True,files=len(value['files']),logical_bytes=sum(v['bytes'] for v in value['files'].values()),
 physical_bytes=value['physical_bytes'],retained_archives=len(value['retained']),future_inputs_verified=len(future),
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''.replace('__SNAPSHOT__', repr(snapshot)), 300))
    for info in roots.values():
        for name, wanted in info['copies'].items(): assert pin(ROOT/'artifacts'/info['local']/'collected'/name) == wanted
    save(OUT/'closed.json', dict(**result, prospective=pin(OUT/'prospective.json'), generator=pin(Path(__file__)),
        scope='Closed generated restore caches and exact VM trace duplicates; every local raw record, canonical model, offline archive and future diagnostic input retained.'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()

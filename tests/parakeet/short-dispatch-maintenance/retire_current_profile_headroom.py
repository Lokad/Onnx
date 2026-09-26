"""Retire only closed VM caches and byte-identical, locally retained outputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/owned-batch-isolation-profile-amd'))
from run import ssh, original, pin, read, write

OUT = ROOT/'artifacts/parakeet-current-profile-headroom-20260925'
FEED = '/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed'
TARGETS = {
    'e5-relocation-tier-diagnostic': {'a-capture','b-capture','c-capture','d-capture','a-export','b-export','c-export','d-export'},
    'parakeet-owned-batch-isolation-root': {'inventory','backend-tests','tensors-tests','logs'},
    'parakeet-owned-batch-isolation-root-policy': {'inventory','backend-tests','backend-tests-256','tensors-tests','tensors-tests-256','logs'},
    'parakeet-owned-batch-isolation-models': {'selected-native-256','selected-native-512','candidate-native-256','candidate-native-512'},
    'parakeet-owned-batch-isolation-shared': {'selected-shared','candidate-shared','selected-e5','candidate-e5'},
}


def main():
    assert not OUT.exists()
    roots = {}
    for name, groups in TARGETS.items():
        folder = ROOT/'artifacts'/(name+'-amd-20260925')
        proof = read(folder/'closed.json')
        if name.endswith('-root'):
            assert not proof['passed'] and proof['preserved_failure']
        else: assert proof['passed']
        for relative, wanted in proof['files'].items(): assert pin(folder/relative) == wanted, relative
        receipt = read(folder/'collected/collection.json')
        assert receipt['terminal']
        copies = {n:v for n,v in receipt['files'].items() if n.split('/')[0] in groups}
        assert copies
        for n,v in copies.items(): assert pin(folder/'collected'/n) == v
        roots['/dev/shm/lokad-'+name+'-20260925'] = dict(local=str(folder), copies=copies,
            cache=name.endswith(('-root','-root-policy')), owners=receipt['identities'],
            code=receipt['code'], collection=pin(folder/'collected/collection.json'), closure=pin(folder/'closed.json'))
    common = original.PRELUDE+'''
def pin(path):
 with Path(path).open('rb') as stream:return dict(bytes=Path(path).stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def read(path):return json.loads(Path(path).read_text())
def live(value):
 try:
  process=psutil.Process(value['pid']);return process.create_time()==value['birth'] and process.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
own=psutil.Process();parents={own.pid,*[p.pid for p in own.parents()]}
for process in psutil.process_iter(['name','cmdline']):
 if process.pid in parents:continue
 assert process.info['name'] not in ['dotnet','perf'],process.info
 assert not (process.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(process.info['cmdline'] or [])),process.info
assert psutil.boot_time()==1789634288.0
roots=__ROOTS__;feed=Path(__FEED__)
protected=set();manifests={}
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
 assert pin(root/'collection.json')==wanted['collection']
 receipt=read(root/'collection.json');assert receipt['terminal'] and receipt['code']==wanted['code']
 assert not any(live(identity) for identity in wanted['owners'])
'''
    common = common.replace('__ROOTS__',repr(roots)).replace('__FEED__',repr(FEED))
    prospective = ssh(common+'''
files={};retained={};physical=0;preserved=[]
for name,wanted in roots.items():
 root=Path(name);paths=[root/n for n in wanted['copies']]
 if wanted['cache']:paths += [p for p in (root/'packages').rglob('*') if p.is_file()]
 for path in paths:
  assert path.resolve()==path and not path.is_symlink() and path.is_relative_to(root)
  if str(path) in protected or path.stat().st_nlink!=1:
   preserved.append(str(path));continue
  identity=pin(path);relative=path.relative_to(root).as_posix()
  if relative in wanted['copies']:assert identity==wanted['copies'][relative]
  else:
   assert path.is_relative_to(root/'packages')
   if path.suffix=='.nupkg':
    archive=(root/'nuget'/path.name) if path.name.startswith('lokad.onnx.') else feed/path.name
    if not archive.exists() and path.name.startswith('lokad.onnx.'):
     archive=root/'nuget/Lokad.Onnx.0.2.0.nupkg'
    assert archive.resolve()==archive and pin(archive)==identity
    retained[str(archive)]=identity
  files[str(path)]=identity;physical+=path.stat().st_blocks*512
assert files and retained
print(json.dumps(dict(passed=True,files=files,retained=retained,manifests=manifests,
 protected_paths=len(protected),preserved=preserved,physical_bytes=physical,
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''')
    OUT.mkdir();write(OUT/'prospective.json',dict(**prospective,roots=roots))
    result = ssh(common+'''
value=__SNAPSHOT__
assert manifests==value['manifests']
for name,wanted in value['files'].items():
 path=Path(name)
 assert path.resolve()==path and not path.is_symlink() and path.stat().st_nlink==1 and name not in protected
 assert any(path.is_relative_to(Path(root)) and
  (path.relative_to(Path(root)).as_posix() in info['copies'] or (info['cache'] and path.is_relative_to(Path(root)/'packages')))
  for root,info in roots.items())
 assert pin(path)==wanted,name
for name,wanted in value['retained'].items():assert name not in value['files'] and pin(name)==wanted
for name in value['files']:Path(name).unlink()
assert all(not Path(name).exists() for name in value['files'])
for name,wanted in value['retained'].items():assert pin(name)==wanted
print(json.dumps(dict(passed=True,files=len(value['files']),physical_bytes=value['physical_bytes'],
 logical_bytes=sum(v['bytes'] for v in value['files'].values()),retained_archives=len(value['retained']),
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''.replace('__SNAPSHOT__',repr(prospective)))
    for info in roots.values():
        for name,wanted in info['copies'].items():assert pin(Path(info['local'])/'collected'/name)==wanted
    write(OUT/'closed.json',dict(**result,prospective=pin(OUT/'prospective.json'),generator=pin(Path(__file__))))
    print(json.dumps(result))


if __name__ == '__main__': main()

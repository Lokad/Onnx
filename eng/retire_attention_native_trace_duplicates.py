"""Retire three terminal VM trace duplicates, preserving local originals and archive."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
import run

BASE = ROOT/'artifacts/parakeet-attention-owned-ort-profile-amd-20260928'
REMOTE = '/dev/shm/lokad-attention-owned-ort-profile-20260928'
OUT = ROOT/'artifacts/parakeet-attention-native-trace-retention-20260928'
pin, read, save = run.pin, run.read, run.write


def main():
    assert not OUT.exists()
    proof = read(BASE/'closed.json')
    assert proof['passed'] and pin(BASE/'closed.json')['sha256'] == '4c6668f50986b24517067b3ff88300e9d3a4096f84b9960223e10be7929048c0'
    assert proof['collection'] == pin(BASE/'collected/collection.json')
    receipt = read(BASE/'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    observation = read(BASE/'collected/profile/observation.json')
    files = {'profile/'+p['file']: {k:p[k] for k in ['bytes','sha256']}
             for p in observation['profiles'].values()}
    assert len(files) == 3
    for name, wanted in files.items():
        assert pin(BASE/'collected'/name) == receipt['files'][name] == wanted
    assert proof['transfer'] == pin(BASE/'transfer.json')
    assert pin(BASE/'results.tar.gz') == read(BASE/'transfer.json')['archive']
    prelude = run.PRELUDE.replace(run.REMOTE, REMOTE)
    common = prelude + f'''
sys.path.insert(0,str(base))
from remote import live,pin,read
assert psutil.boot_time()==1789634288.0
me=psutil.Process();ancestors={{me.pid,*[p.pid for p in me.parents()]}}
for process in psutil.process_iter(['pid','name','cmdline']):
 if process.pid in ancestors:continue
 assert process.info['name'] not in ['dotnet','perf'],process.info
 assert not (process.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(process.info['cmdline'] or [])),process.info
assert base.resolve()==base and base.parent==Path('/dev/shm')
assert read(base/'state.json')['complete'] and read(base/'state.json')['code']==0
assert all(not live(i) for i in {proof['terminal_owners']!r})
assert pin(base/'collection.json')=={proof['collection']!r}
files={files!r}
from functools import lru_cache
target_ids={{((base/n).stat().st_dev,(base/n).stat().st_ino) for n in files}}
assert len(target_ids)==3
@lru_cache(maxsize=65536)
def references_target(name):
 try:
  info=Path(name).stat()
  return (info.st_dev,info.st_ino) in target_ids
 except FileNotFoundError:return False
assert all(references_target(str(base/n)) for n in files)
assert not references_target(str(base/'remote.py'))
protected=[];manifests={{}}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  path=folder/name
  if not path.exists():continue
  value=read(path);manifests[str(path)]=pin(path)
  for referenced in [*(str(folder/n) for n in value.get('files',{{}})),*value.get('external',{{}})]:
   if references_target(referenced):protected.append(dict(manifest=str(path),input=referenced))
assert not protected,protected
for name,wanted in files.items():
 path=base/name
 assert path.resolve()==path and path.parent==base/'profile' and path.suffix=='.json'
 assert not path.is_symlink() and path.stat().st_nlink==1
 assert pin(path)==wanted
'''
    prospective = run.ssh(common + '''
print(json.dumps(dict(passed=True,manifests=manifests,
 allocated=sum((base/n).stat().st_blocks*512 for n in files),
 available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)))
''')
    OUT.mkdir()
    save(OUT/'prepared.json', dict(**prospective, closure=pin(BASE/'closed.json'),
         files=files, source=pin(Path(__file__)), archive=pin(BASE/'results.tar.gz'),
         policy='Only unreferenced terminal VM copies are removed; complete local traces and their archive remain immutable.',
         initial_read_only_refusals=[
             'The native helper has no idle function; the import failed before any mutation. Use the existing offload worker process-idle guard.',
             'Repeated path resolution exceeded the read-only SSH limit. Remote inspection confirmed no remaining recent worker and all three traces present. Compare cached resolved file identities instead, retaining every manifest and input check.']))
    result = run.ssh(common + f'''
assert manifests=={prospective['manifests']!r}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
physical=sum((base/n).stat().st_blocks*512 for n in files)
assert physical=={prospective['allocated']!r}
for name in files:(base/name).unlink()
assert all(not (base/name).exists() for name in files)
print(json.dumps(dict(passed=True,files=len(files),allocated_reclaimed=physical,before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free))))
''')
    for name, wanted in files.items():assert pin(BASE/'collected'/name) == wanted
    assert pin(BASE/'results.tar.gz') == read(BASE/'transfer.json')['archive']
    save(OUT/'closed.json', dict(**result, preparation=pin(OUT/'prepared.json'),
         all_local_originals_and_archive_retained=True))
    print(json.dumps(result))


if __name__ == '__main__': main()

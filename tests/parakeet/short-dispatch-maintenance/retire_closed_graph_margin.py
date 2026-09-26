"""Retire bounded VM-only tensor copies with closed local evidence and no live work."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-current-graphs-amd'))
import run
from protocol import pin,read,save

OUT=ROOT/'artifacts/parakeet-pad-graph-headroom-retention-20260926'
TARGETS={
 'parakeet-owned-batch-isolation-shared-amd-20260925':'4a8bb671dcfef2ef0193ca9105ab07023aedc8d2cba8cef0c578981769dff423',
 'parakeet-owned-batch-isolation-pyannote-amd-20260925':'d5f36fc9dd5cbce573072556e80146860b26133d271b24a7da9ce70b6c6e86a9',
 'parakeet-packed-final-row-pyannote-amd-20260925':'8471ed0c4f9cdca7ba32edf2ba7ef744bedae4c0b0b037af52f1c87642230595',
 'parakeet-pad-current-pyannote-amd-20260926':'5e2b54bb685d9e954d3840524c486b55a22aadc5a574b6cc189bedf5bf487435',
 'parakeet-owned-batch-isolation-graphs-amd-20260925':'def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638',
 'parakeet-packed-final-row-graphs-amd-20260925':'0b82805aa6de28287c9bc9242416b3abd5c58b76fdff57a0416208f29f3707f2',
 'parakeet-observed-dense-where-pyannote-amd-20260924':'e36fe9c83405608659ebdc4d673c185c8805a5414a6b01d401904209d59417dd'}


def main():
    assert not OUT.exists();roots={}
    for name,digest in TARGETS.items():
        folder=ROOT/'artifacts'/name;assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');receipt=read(folder/'collected/collection.json')
        assert proof['passed'] and proof['files']['collected/collection.json']==pin(folder/'collected/collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        copies={n:v for n,v in receipt['files'].items() if '/output/' in n and Path(n).suffix in ['.f32','.bin']}
        assert copies
        for n,wanted in copies.items():assert pin(folder/'collected'/n)==wanted
        roots['/dev/shm/lokad-'+name.replace('-amd-','-')]=dict(local=name,copies=copies,owners=receipt['identities'],receipt=pin(folder/'collected/collection.json'))
    assert sum(len(v['copies']) for v in roots.values())==1118
    OUT.mkdir();save(OUT/'prospective.json',dict(roots=roots,helper=pin(__file__)))
    # Use only the transport prelude; the new graph namespace is not created.
    script=run.PRELUDE+f'''
roots={roots!r}
import hashlib
def pin(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert psutil.boot_time()==1789634288.0 and not base.exists()
ancestors={{os.getpid(),*[p.pid for p in psutil.Process().parents()]}}
for p in psutil.process_iter(['name','cmdline']):
 if p.pid in ancestors:continue
 assert p.info['name'] not in ['dotnet','perf']
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
def live(i):
 try:
  p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
paths=set();manifests={{}}
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  p=folder/name
  if not p.is_file():continue
  value=json.loads(p.read_text());manifests[str(p)]=pin(p)
  paths.update(str(folder/n) for n in value.get('files',{{}}));paths.update(value.get('external',{{}}))
  for link in value.get('links',{{}}).values():
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
ledger=Path('/dev/shm/lokad-pad-graph-output-retention-20260926.json');assert not ledger.exists()
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
ledger.write_text(json.dumps(dict(prospective=True,targets=selected,manifests=manifests)))
for name in selected:Path(name).unlink()
assert all(not Path(name).exists() for name in selected)
for name,wanted in manifests.items():assert pin(name)==wanted
value=dict(passed=True,files=len(selected),physical_bytes=sum(v['physical'] for v in selected.values()),
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 protected_inputs_preserved=True,all_owners_terminal=True,no_inference_during_cleanup=True,targets=selected,preserved=preserved,manifests=manifests)
ledger.write_text(json.dumps(value))
print(json.dumps(value))
'''
    value=json.loads(run.ssh(script,300))
    for info in roots.values():
        for n,wanted in info['copies'].items():assert pin(ROOT/'artifacts'/info['local']/'collected'/n)==wanted
    save(OUT/'closed.json',dict(**value,all_local_bytes_retained=True,prospective=pin(OUT/'prospective.json')))
    print(json.dumps({k:v for k,v in value.items() if k not in ['targets','preserved','manifests']}))


if __name__=='__main__':main()

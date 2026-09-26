"""Retire verified closed VM diagnostic output copies, retaining authoritative local bytes."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/pad-memory-diagnostic-amd'))
from run import ssh,PRELUDE
from protocol import pin,read,save

BASE=ROOT/'artifacts/parakeet-pad-memory-output-retention-20260926'
TARGETS=[
 ('parakeet-pad-memory-diagnostic-amd-20260926','parakeet-pad-memory-diagnostic-20260926','51e714628218f2908ac46d0e8336b29003d1f7a266e424bd37ec946fa1c7589b'),
 ('parakeet-pad-warmup-diagnostic-amd-20260926','parakeet-pad-warmup-diagnostic-20260926','4402da8effb9467d71fd6472678ead08b714ba147d9e3fe3f8788bb715ec9b4f'),
 ('parakeet-pad-application-diagnostic-amd-20260926','parakeet-pad-application-diagnostic-20260926','0f7c5cd1c4d022aaf72ffc1c8f8a764336e65bad171d0435e5f7892871004020'),
 ('parakeet-pad-current-screen-amd-20260926','parakeet-pad-current-screen-20260926','5b8d7df749d600238dfc6e1c9ead50486b0027754656c57d0c553573ada1b17d'),
 ('parakeet-pad-runtime-diagnostic-amd-20260923','parakeet-pad-runtime-diagnostic-20260923','2e718d3095eaea8ec79fb263d5ca504564a5e5a83e986ef5535c53505b3864de')]

COMMON=PRELUDE+'''
from protocol import pin,read
from remote import idle,live
idle();assert psutil.boot_time()==1789634288.0
manifests={};paths=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 for name in ['payload.json','stage.json','spec.json']:
  p=folder/name
  if not p.is_file():continue
  value=read(p);manifests[str(p)]=pin(p)
  paths.update(str(folder/n) for n in value.get('files',{}))
  paths.update(value.get('external',{}))
  for link in value.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):paths.add(link['source'])
protected={str(Path(p).resolve()) for p in paths}
'''


def inventory():
    assert not BASE.exists()
    roots={}
    for local,remote,digest in TARGETS:
        folder=ROOT/'artifacts'/local
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');receipt=read(folder/'collected/collection.json')
        assert proof['files']['collected/collection.json']==pin(folder/'collected/collection.json')
        assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
        copies={}
        for name,wanted in receipt['files'].items():
            path=Path(name)
            if path.parts[0].startswith(('current-','candidate-')) and wanted['bytes']>=65536 and \
                    (path.suffix in ('.json','.jsonl','.nettrace') or path.name.endswith('.jsonl.gz')):
                assert pin(folder/'collected'/name)==wanted
                copies[name]=wanted
        roots['/dev/shm/lokad-'+remote]=dict(local=local,closure=pin(folder/'closed.json'),
            receipt=pin(folder/'collected/collection.json'),owners=receipt['identities'],copies=copies)
    BASE.mkdir();save(BASE/'prospective.json',dict(roots=roots,source=pin(__file__)))
    value=json.loads(ssh(COMMON+f'''
roots={roots!r};eligible={{}};preserved=[]
for name,info in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'collection.json')==info['receipt']
 assert not any(live(i) for i in info['owners'])
 for relative,wanted in info['copies'].items():
  p=root/relative
  if not p.exists() or p.is_symlink() or str(p.resolve()) in protected or p.stat().st_nlink!=1:
   preserved.append(str(p));continue
  assert p.resolve()==p and p.is_relative_to(root) and pin(p)==wanted
  eligible[str(p)]=dict(**wanted,physical=p.stat().st_blocks*512,root=name,relative=relative)
idle()
print(json.dumps(dict(passed=True,removed=False,eligible=eligible,preserved=preserved,manifests=manifests,
 available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)))
''',300))
    save(BASE/'eligible.json',value)
    print(json.dumps(dict(files=len(value['eligible']),physical=sum(v['physical'] for v in value['eligible'].values()),
        roots={name:sum(v['physical'] for v in value['eligible'].values() if v['root']==name) for name in roots},
        available=value['available'],tmpfs=value['tmpfs'])))


def retire():
    assert not (BASE/'closed.json').exists()
    prospective=read(BASE/'prospective.json');assert prospective['source']==pin(__file__)
    selected=read(BASE/'eligible.json');assert selected['passed'] and not selected['removed']
    for name,row in selected['eligible'].items():
        root=prospective['roots'][row['root']]
        local=ROOT/'artifacts'/root['local']/'collected'/row['relative']
        assert pin(local)=={k:row[k] for k in ('bytes','sha256')}
    value=json.loads(ssh(COMMON+f'''
roots={prospective['roots']!r};selected={selected['eligible']!r}
for name,wanted in {selected['manifests']!r}.items():assert pin(name)==wanted
for root,info in roots.items():
 assert not any(live(i) for i in info['owners'])
 assert pin(Path(root)/'collection.json')==info['receipt']
for name,row in selected.items():
 p=Path(name);root=Path(row['root'])
 assert root.parent==Path('/dev/shm') and p.resolve()==p and p.is_relative_to(root)
 assert not p.is_symlink() and p.stat().st_nlink==1 and name not in protected
 assert pin(p)=={{k:row[k] for k in ('bytes','sha256')}}
idle();before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in selected:Path(name).unlink()
assert all(not Path(name).exists() for name in selected)
print(json.dumps(dict(passed=True,removed=True,files=len(selected),physical=sum(v['physical'] for v in selected.values()),
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 all_owners_terminal=True,protected_inputs_preserved=True)))
''',300))
    for name,row in selected['eligible'].items():
        root=prospective['roots'][row['root']]
        assert pin(ROOT/'artifacts'/root['local']/'collected'/row['relative'])=={k:row[k] for k in ('bytes','sha256')}
    save(BASE/'closed.json',dict(**value,prospective=pin(BASE/'prospective.json'),eligible=pin(BASE/'eligible.json'),all_local_bytes_retained=True))
    print(json.dumps(value))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('inventory','retire')
    globals()[sys.argv[1]]()

"""Retire terminal VM copies retained exactly; preserve external inputs and products."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/vm-retention-lstm-runtime-v2-20260927'
sys.path.insert(0, str(ROOT/'tests/parakeet/decoder-lstm-runtime-observation'))
import run


def pin(path):
    with path.open('rb') as stream: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
def read(path): return json.loads(path.read_text())
def save(path, value):
    with path.open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')


REMOTE_COMMON = r'''
from pathlib import Path
import hashlib,json,os,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
os.sched_setaffinity(0,{0})
own=psutil.Process();ancestors={own.pid,*[p.pid for p in own.parents()]}
for p in psutil.process_iter(['pid','name','cmdline']):
 if p.pid in ancestors:continue
 assert p.info['name'] not in ['dotnet','perf']
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
assert psutil.boot_time()==1789634288.0
def pin(p):
 with Path(p).open('rb') as f:return dict(bytes=Path(p).stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def terminal(receipt):
 assert receipt['terminal'] and receipt.get('input_error') is None
 for i in receipt['identities']:
  try:
   p=psutil.Process(i['pid']);assert p.create_time()!=i['birth'] or p.status()==psutil.STATUS_ZOMBIE
  except psutil.NoSuchProcess:pass
protected=set();external=set();payloads={}
for path in Path('/dev/shm').glob('lokad-*/payload.json'):
 value=json.loads(path.read_text());payloads[str(path)]=pin(path)
 protected.update(os.path.normpath(str(path.parent/name)) for name in value.get('files',{}))
 external.update(os.path.normpath(name) for name in value.get('external',{}))
protected.update(external)
keep={'runtime','runtimes','products','built','fixtures','assets','parakeet-reference','models','nuget-feed','tracer','export-runtime'}
def eligible(folder,name):
 p=folder/name
 if Path(name).is_absolute() or '..' in Path(name).parts or len(Path(name).parts)<2:return False
 if Path(name).parts[0] in keep or str(p) in external or not p.is_file() or p.is_symlink():return False
 # An obsolete stage's own source snapshot may be retired only when its exact
 # bytes were collected locally and no other campaign references this path.
 if str(p) in protected and Path(name).parts[0]!='source':return False
 return p.stat().st_nlink==1 and p.stat().st_size>=4096
'''


def inventory():
    assert not OUT.exists()
    roots = {}
    for path in (ROOT/'artifacts').glob('*/collected/collection.json'):
        folder = path.parents[1]; receipt = read(path)
        closure = next((p for p in [folder/'closed.json', folder/'failed.json'] if p.exists()), None)
        if closure is None or not receipt.get('terminal') or receipt.get('input_error') is not None: continue
        proof = read(closure)
        if proof.get('files', {}).get('collected/collection.json') != pin(path): continue
        identity = pin(path)
        roots.setdefault(identity['sha256'], dict(local=folder.relative_to(ROOT).as_posix(), receipt=identity, closure=pin(closure)))
    script = REMOTE_COMMON+'\nROOTS='+repr(roots)+r'''
rows=[];matched={}
for path in Path('/dev/shm').glob('lokad-*/collection.json'):
 identity=pin(path)
 if identity['sha256'] not in ROOTS:continue
 info=ROOTS[identity['sha256']];folder=path.parent;root=str(folder)
 assert identity==info['receipt']
 receipt=json.loads(path.read_text());terminal(receipt);matched[root]=info
 for name,wanted in receipt['files'].items():
  if not eligible(folder,name):continue
  p=folder/name;stat=p.stat()
  if stat.st_size!=wanted['bytes']:continue
  rows.append(dict(root=root,name=name,wanted=wanted,allocated=stat.st_blocks*512,device=stat.st_dev,inode=stat.st_ino))
print(json.dumps(dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free,
 roots=matched,candidates=sorted(rows,key=lambda r:r['allocated'],reverse=True),payloads=payloads,protected_paths=len(protected))))
'''
    value = json.loads(run.ssh(script, 180))
    OUT.mkdir(); save(OUT/'inventory.json', value)
    print(json.dumps(dict(roots=len(value['roots']), candidates=len(value['candidates']),
        candidate_allocated=sum(r['allocated'] for r in value['candidates']), available=value['available'], tmpfs=value['tmpfs'])))


def plan():
    assert not (OUT/'plan.json').exists()
    value = read(OUT/'inventory.json'); extra = read(OUT/'supplement.json'); selected = []; excluded = []; allocated = 0
    assert extra['payloads'] == value['payloads']
    for row in sorted(value['candidates']+extra['candidates'], key=lambda r:r['allocated'], reverse=True):
        folder = ROOT/value['roots'][row['root']]['local']
        source = (ROOT/row['local']).resolve() if 'local' in row else (folder/'collected'/row['name']).resolve()
        assert source.is_relative_to((ROOT/'artifacts').resolve())
        actual = pin(source) if source.is_file() else None
        if actual != row['wanted']:
            excluded.append(dict(root=row['root'], name=row['name'], actual=actual, reason='Exact local copy unavailable; preserve VM file'))
            continue
        selected.append(dict(row, local=source.relative_to(ROOT).as_posix()))
        allocated += row['allocated']
        if allocated >= 1024**3: break
    assert selected
    assert len({(r['root'], r['name']) for r in selected}) == len(selected)
    assert len({(r['device'], r['inode']) for r in selected}) == len(selected)
    save(OUT/'plan.json', dict(source=pin(Path(__file__)), inventory=pin(OUT/'inventory.json'), supplement=pin(OUT/'supplement.json'),
        retired_files=selected, expected_allocated_reclaimed=allocated,
        excluded=excluded,
        preservation='Every byte retained locally; all externally referenced inputs, runtimes, products, models and provenance kept. Only single-link terminal copies, including obsolete collected source snapshots, are eligible. Old source stages cannot be replayed.'))
    print(json.dumps(dict(files=len(selected), expected_allocated_reclaimed=allocated, roots=len({r['root'] for r in selected}), excluded=len(excluded))))


def supplement():
    assert not (OUT/'supplement.json').exists()
    folder = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
    assert pin(folder/'failed.json')['sha256'] == '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'
    root = '/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927'
    inventory = read(OUT/'inventory.json'); assert root in inventory['roots']
    extras = []
    for name, wanted in read(folder/'failed.json')['files'].items():
        if name.startswith('collected/packages/') and wanted['bytes'] >= 4096:
            path = folder/name; assert pin(path) == wanted
            extras.append(dict(root=root, name=name.removeprefix('collected/'), wanted=wanted, local=path.relative_to(ROOT).as_posix()))
    script = REMOTE_COMMON+'\nEXTRAS='+repr(extras)+r'''
rows=[]
for row in EXTRAS:
 folder=Path(row['root']);p=folder/row['name']
 if not eligible(folder,row['name']) or pin(p)!=row['wanted']:continue
 stat=p.stat();rows.append(dict(**row,allocated=stat.st_blocks*512,device=stat.st_dev,inode=stat.st_ino))
print(json.dumps(dict(candidates=rows,payloads=payloads)))
'''
    result = json.loads(run.ssh(script, 180)); assert result['payloads'] == inventory['payloads']
    save(OUT/'supplement.json', result)
    print(json.dumps(dict(private_cache_files=len(result['candidates']), allocated=sum(r['allocated'] for r in result['candidates']))))


def retire():
    assert not (OUT/'retired.json').exists() and not (OUT/'started.json').exists()
    wanted = read(OUT/'plan.json'); inventory = read(OUT/'inventory.json')
    assert wanted['source'] == pin(Path(__file__)) and wanted['inventory'] == pin(OUT/'inventory.json')
    assert wanted['supplement'] == pin(OUT/'supplement.json')
    for row in wanted['retired_files']: assert pin(ROOT/row['local']) == row['wanted'], row['local']
    script = REMOTE_COMMON+'\nPLAN='+repr(wanted)+'\nINVENTORY='+repr(inventory)+r'''
assert payloads==INVENTORY['payloads'],'Protected-input census changed'
roots={r['root'] for r in PLAN['retired_files']}
for root in roots:
 assert Path(root).parent==Path('/dev/shm') and Path(root).name.startswith('lokad-')
 path=Path(root)/'collection.json';assert pin(path)==INVENTORY['roots'][root]['receipt'];terminal(json.loads(path.read_text()))
for row in PLAN['retired_files']:
 folder=Path(row['root']);p=(folder/row['name']).resolve()
 assert p.is_relative_to(folder.resolve()) and eligible(folder,row['name'])
 stat=p.stat();assert (stat.st_dev,stat.st_ino,stat.st_blocks*512)==(row['device'],row['inode'],row['allocated'])
 assert pin(p)==row['wanted'],str(p)
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in PLAN['retired_files']:(Path(row['root'])/row['name']).unlink()
assert all(not (Path(r['root'])/r['name']).exists() for r in PLAN['retired_files'])
assert all(pin(path)==wanted for path,wanted in payloads.items())
after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
print(json.dumps(dict(passed=True,files=len(PLAN['retired_files']),before=before,after=after,expected_allocated_reclaimed=PLAN['expected_allocated_reclaimed'],payloads_unchanged=len(payloads))))
'''
    save(OUT/'started.json', dict(plan=pin(OUT/'plan.json'), driver_sha256=hashlib.sha256(script.encode()).hexdigest()))
    result = json.loads(run.ssh(script, 180)); save(OUT/'retired.json', result)
    for row in wanted['retired_files']: assert pin(ROOT/row['local']) == row['wanted'], row['local']
    print(json.dumps(result))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['inventory', 'supplement', 'plan', 'retire']
    globals()[sys.argv[1]]()

"""One-time retirement of verified closed outputs and the qualified private cache."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-decoder-model-headroom-20260927'
PROOF = ROOT/'artifacts/parakeet-decoder-models-local-output-proof-v2-20260927.json'
QUALIFIED = ROOT/'artifacts/parakeet-rational-sigmoid-root-amd-20260927'


def pin(path):
    with path.open('rb') as f: return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text())


def main():
    assert not OUT.exists(), 'Never replay a completed or partially attempted retirement'
    assert pin(PROOF)['sha256'] == '3071a132fc75d782a590d234991d9c1c20806318098435332a78064ca279384d'
    proof = read(PROOF); assert proof['passed'] and len(proof['verified']) == 296
    outputs = proof['verified']; closed = {}
    for row in outputs:
        path = ROOT/row['local_closed']
        assert pin(path) == row['closed_identity']
        c = closed.setdefault(str(path), read(path)); assert c.get('passed') or c.get('preserved_failure')
        receipt_path = ROOT/row['local_collection']; assert pin(receipt_path) == row['receipt']
        relative = receipt_path.relative_to(path.parent).as_posix()
        bound = c.get('files', {}).get(relative) == row['receipt'] or c.get('collection') == row['receipt']
        for link in row['review_links']:
            assert pin(ROOT/link['path']) == link['identity'] == c['build_review']
            review = read(ROOT/link['path']); assert review['passed']
            bound |= review.get('collection') == row['receipt']
        assert bound
        receipt = read(receipt_path)
        assert receipt['terminal'] and receipt['code'] == 0 and receipt['files'][row['relative']] == row['identity']
        assert pin(ROOT/row['local']) == row['identity']
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'f6df53ce2ab773898bc84f144abeda97908d383fef9a4df4b7299a22a4d3594d'
    qualified = read(QUALIFIED/'closed.json'); assert qualified['passed'] and qualified['remote_terminal']
    receipt_path = QUALIFIED/'collected/collection.json'
    assert pin(receipt_path) == qualified['files']['collected/collection.json']
    package_path = QUALIFIED/'collected/nuget/Lokad.Onnx.0.2.0.nupkg'
    assert pin(package_path) == read(receipt_path)['files']['nuget/Lokad.Onnx.0.2.0.nupkg']
    cache = dict(base='/dev/shm/lokad-parakeet-rational-sigmoid-root-20260927',
        collection=pin(receipt_path), package=pin(package_path))
    script = REMOTE_SCRIPT.replace('@@OUTPUTS@@', repr(outputs)).replace('@@CACHE@@', repr(cache))
    compile(script, 'decoder-model-duplicate-retirement', 'exec')
    OUT.mkdir()
    def save(name, value):
        with (OUT/name).open('x') as f: json.dump(value, f, indent=2); f.write('\n')
    save('intent.json', dict(tool=pin(Path(__file__)), proof=pin(PROOF), outputs=outputs, cache=cache))
    with (OUT/'remote.py').open('x', newline='\n') as f: f.write(script)
    loader = importlib.util.spec_from_file_location('retirement_transport', ROOT/'tests/parakeet/ort-diagnosis-amd/run.py')
    transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
    result = transport.ssh(script)
    save('result.json', result)
    assert result['passed'] and result['output_files'] == 296
    # Local evidence remains complete after VM retirement.
    for row in outputs: assert pin(ROOT/row['local']) == row['identity']
    save('closed.json', dict(passed=True, result=pin(OUT/'result.json'), intent=pin(OUT/'intent.json'),
        script=pin(OUT/'remote.py'), local_outputs_verified=296, remote_owners_terminal=True))
    print(json.dumps({k: v for k, v in result.items() if k not in ['files', 'retained_archives']}))


REMOTE_SCRIPT = r'''
from pathlib import Path
import hashlib,json,os,signal,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
os.sched_setaffinity(0,{0});signal.alarm(110)
def pin(path):
 with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(path):return json.loads(path.read_text())
def live(i):
 try:
  p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False
own=psutil.Process();ancestors={own.pid,*[p.pid for p in own.parents()]}
assert psutil.boot_time()==1789634288.0
for p in psutil.process_iter(['name','cmdline']):
 if p.pid in ancestors:continue
 assert p.info['name'] not in ['dotnet','perf']
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
unresolved=set();documents=0
for folder in Path('/dev/shm').glob('lokad-*'):
 if not folder.is_dir():continue
 for name in ['spec.json','payload.json','stage.json']:
  path=folder/name
  if not path.is_file():continue
  value=read(path);documents+=1
  unresolved.update(str(folder/n) for n in value.get('files',{}));unresolved.update(value.get('external',{}))
  for link in value.get('links',{}).values():
   if isinstance(link,dict) and isinstance(link.get('source'),str):unresolved.add(link['source'])
protected={str(Path(n).resolve()) for n in unresolved};checked=[];receipts={}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in @@OUTPUTS@@:
 path=Path(row['path']);receipt_path=Path(row['collection']);base=receipt_path.parent.resolve()
 assert base.parent==Path('/dev/shm') and path==base/row['relative'] and path.resolve()==path and path.is_relative_to(base)
 assert not path.is_symlink() and str(path) not in protected
 if str(receipt_path) not in receipts:
  assert pin(receipt_path)==row['receipt'];receipt=read(receipt_path)
  assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt.get('identities',[]))
  receipts[str(receipt_path)]=receipt
 assert pin(path)==row['identity']==receipts[str(receipt_path)]['files'][row['relative']]
 st=path.stat();assert st.st_nlink==1 and st.st_blocks*512==row['physical']
 checked.append(dict(path=str(path),identity=row['identity'],physical=st.st_blocks*512,retained_locally=row['local']))
cache=@@CACHE@@;base=Path(cache['base']).resolve();target=base/'packages'
assert base.parent==Path('/dev/shm') and target.resolve()==target and target.is_dir() and not target.is_symlink()
receipt_path=base/'collection.json';assert pin(receipt_path)==cache['collection'];receipt=read(receipt_path)
assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt['identities'])
feed=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed')
archives={p.name.lower():p for p in feed.glob('*.nupkg')}
package=base/'nuget/Lokad.Onnx.0.2.0.nupkg';assert pin(package)==cache['package'];archives[package.name.lower()]=package
retained=[]
for directory in target.glob('*/*'):
 assert directory.is_dir() and not directory.is_symlink()
 found=list(directory.glob('*.nupkg'));assert len(found)==1
 private=found[0];canonical=archives[private.name.lower()]
 assert not canonical.resolve().is_relative_to(target) and pin(private)==pin(canonical)
 retained.append(dict(private=str(private),canonical=str(canonical),identity=pin(canonical)))
assert len(retained)==24
for path in target.rglob('*'):
 assert not path.is_symlink() and path.resolve().is_relative_to(target)
 if path.is_file():
  assert str(path) not in protected
  st=path.stat();assert st.st_nlink==1
  checked.append(dict(path=str(path),identity=pin(path),physical=st.st_blocks*512,private_cache=True))
assert len({r['path'] for r in checked})==len(checked) and len(checked)==921
# Finish all ownership, path, reference and byte checks before the first unlink.
for row in checked:
 path=Path(row['path']);assert pin(path)==row['identity'];path.unlink()
for path in sorted((p for p in target.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.rmdir()
target.rmdir()
for row in retained:assert pin(Path(row['canonical']))==row['identity']
print(json.dumps(dict(passed=True,files=checked,output_files=296,cache_files=625,retained_archives=retained,
 protected_paths=len(protected),frozen_documents=documents,physical_bytes_freed=sum(r['physical'] for r in checked),
 before=before,after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''


if __name__ == '__main__': main()
